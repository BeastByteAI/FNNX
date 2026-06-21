"""End-to-end tests for the MLflow → FNNX converter across the gradient-boosting
framework matrix: XGBoost, LightGBM, and CatBoost.

For each framework: train a tiny classifier (fixed seed, ~20 rows, few
estimators), save with the matching ``mlflow.<flavor>`` flavor and an inferred
signature, package via ``package_mlflow_model``, load with ``fnnx.Runtime``, and
assert the round-trip predictions match the original ``model.predict`` exactly.
Also asserts ``input_mode="columns"``, env pins the framework, and that no
self-containment warning is emitted (native, self-describing artifacts).
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import unittest
from unittest import mock

import pytest


pytest.importorskip("mlflow")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")


def _make_training_data():
    """Return ``(x_df, y)`` — 20-row binary classification frame, fixed seed."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 20
    a = rng.random(n)
    x = pd.DataFrame(
        {
            "a": a,
            "b": rng.random(n),
            "c": rng.random(n),
        }
    )
    # Deterministic, separable labels driven by column `a`.
    y = (a > 0.5).astype(int).tolist()
    return x, y


def _train_xgboost():
    import xgboost as xgb  # type: ignore[import-not-found]

    x, y = _make_training_data()
    model = xgb.XGBClassifier(n_estimators=3, max_depth=2, random_state=0)
    model.fit(x, y)
    return model, x


def _train_lightgbm():
    import lightgbm as lgb  # type: ignore[import-not-found]

    x, y = _make_training_data()
    model = lgb.LGBMClassifier(
        n_estimators=3,
        max_depth=2,
        random_state=0,
        min_data_in_leaf=1,
        min_data_in_bin=1,
        verbose=-1,
    )
    model.fit(x, y)
    return model, x


def _train_catboost():
    import catboost as cb  # type: ignore[import-not-found]

    x, y = _make_training_data()
    model = cb.CatBoostClassifier(
        iterations=3,
        depth=2,
        random_seed=0,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(x, y)
    return model, x


# (flavor name, trainer fn, importorskip target, expected loader_module,
#  framework package prefix in requirements.txt).
_FRAMEWORK_MATRIX = [
    ("xgboost", _train_xgboost, "xgboost", "mlflow.xgboost", "xgboost"),
    ("lightgbm", _train_lightgbm, "lightgbm", "mlflow.lightgbm", "lightgbm"),
    ("catboost", _train_catboost, "catboost", "mlflow.catboost", "catboost"),
]


class TestGradientBoostingFrameworks(unittest.TestCase):
    """Each gradient-boosting framework round-trips through FNNX losslessly."""

    def _run_framework(
        self,
        flavor: str,
        trainer,
        loader_module: str,
        framework_pkg: str,
    ) -> None:
        pytest.importorskip(flavor)
        import mlflow  # type: ignore[import-not-found]
        import numpy as np
        from mlflow.models.signature import infer_signature  # type: ignore[import-not-found]

        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        model, x = trainer()
        save_fn = getattr(mlflow, flavor).save_model

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, f"{flavor}_model")
            sig = infer_signature(x, model.predict(x))
            save_fn(model, model_dir, signature=sig)

            out = os.path.join(tmp, f"{flavor}.fnnx")
            with mock.patch("fnnx.extras.mlflow.console.warn") as warn_mock:
                package_mlflow_model(model_dir, out)

            # Inspect the produced package.
            with tarfile.open(out, "r") as tar:
                manifest = json.loads(
                    tar.extractfile("manifest.json").read().decode()  # type: ignore[union-attr]
                )
                variant_config = json.loads(
                    tar.extractfile("variant_config.json").read().decode()  # type: ignore[union-attr]
                )
                env = json.loads(
                    tar.extractfile("env.json").read().decode()  # type: ignore[union-attr]
                )
                meta = json.loads(
                    tar.extractfile("meta.json").read().decode()  # type: ignore[union-attr]
                )

            cfg = variant_config["extra_values"]["fnnx_mlflow"]
            self.assertEqual(
                cfg["input_mode"],
                "columns",
                f"{flavor}: expected columns mode, got {cfg['input_mode']!r}",
            )
            self.assertEqual(cfg["column_order"], ["a", "b", "c"])

            # Manifest: one NDJSON input per column with Array[float64].
            self.assertEqual(len(manifest["inputs"]), 3)
            for spec in manifest["inputs"]:
                self.assertEqual(spec["content_type"], "NDJSON")
                self.assertTrue(spec["dtype"].startswith("Array[float64]"))

            # Env pins the framework + mlflow + fnnx.
            deps = [
                d["package"]
                for d in env["python3::conda_pip"]["dependencies"]
            ]
            self.assertTrue(
                any(d.startswith(framework_pkg) for d in deps),
                f"{flavor}: expected {framework_pkg} in env deps, got {deps!r}",
            )
            self.assertTrue(any(d.startswith("mlflow") for d in deps))
            self.assertTrue(any(d.startswith("fnnx") for d in deps))

            # Native flavor → no self-containment warning emitted.
            warned = [c.args[0] for c in warn_mock.call_args_list]
            for msg in warned:
                self.assertNotIn(
                    "serializes Python objects by reference", msg
                )
                self.assertNotIn("referenced from the Hugging Face Hub", msg)

            # Meta records the right loader.
            self.assertEqual(
                meta[0]["payload"]["loader_module"], loader_module
            )

            # Round-trip through fnnx.Runtime: predictions match exactly.
            rt = Runtime(out)
            inputs = {
                "a": np.asarray(x["a"].values, dtype=np.float64),
                "b": np.asarray(x["b"].values, dtype=np.float64),
                "c": np.asarray(x["c"].values, dtype=np.float64),
            }
            result = rt.compute(inputs, {})

        self.assertIn("predictions", result)
        expected = model.predict(x).tolist()
        self.assertEqual(
            result["predictions"],
            expected,
            f"{flavor}: round-trip predictions diverged from native model.predict",
        )

    def test_xgboost_round_trip(self) -> None:
        flavor, trainer, _, loader, pkg = _FRAMEWORK_MATRIX[0]
        self._run_framework(flavor, trainer, loader, pkg)

    def test_lightgbm_round_trip(self) -> None:
        flavor, trainer, _, loader, pkg = _FRAMEWORK_MATRIX[1]
        self._run_framework(flavor, trainer, loader, pkg)

    def test_catboost_round_trip(self) -> None:
        flavor, trainer, _, loader, pkg = _FRAMEWORK_MATRIX[2]
        self._run_framework(flavor, trainer, loader, pkg)


if __name__ == "__main__":
    unittest.main()
