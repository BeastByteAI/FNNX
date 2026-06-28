"""End-to-end test for ``package_mlflow_model``.

Trains a tiny deterministic sklearn model, packages it with the converter, and
asserts a round-trip through ``fnnx.Runtime`` reproduces the original
``model.predict`` output.

Also exercises the local-directory path without ``mlflow`` installed (the
packaging step must not import mlflow), and that a non-local URI under the
same condition raises ``ImportError``.
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import unittest

import pytest


pytest.importorskip("mlflow")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")


def _train_tiny_sklearn():
    """Return ``(model, x_df, y)`` for a deterministic 4-row binary classifier."""
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-not-found]
    import pandas as pd

    x = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [3.0, 2.0, 1.0, 0.0],
        }
    )
    y = [0, 1, 0, 1]
    model = RandomForestClassifier(n_estimators=3, random_state=0)
    model.fit(x, y)
    return model, x, y


def _save_sklearn_model(tmp: str):
    """Save a tiny sklearn model and return ``(model_dir, model, x_df)``."""
    import mlflow  # type: ignore[import-not-found]
    from mlflow.models.signature import infer_signature  # type: ignore[import-not-found]

    model, x, _ = _train_tiny_sklearn()
    model_dir = os.path.join(tmp, "mlflow_model")
    sig = infer_signature(x, model.predict(x))
    mlflow.sklearn.save_model(model, model_dir, signature=sig)  # type: ignore[attr-defined]
    return model_dir, model, x


class TestSklearnRoundTrip(unittest.TestCase):
    def test_columns_mode_round_trip(self):
        import numpy as np

        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            model_dir, model, x = _save_sklearn_model(tmp)
            out = os.path.join(tmp, "m.fnnx")
            package_mlflow_model(model_dir, out)

            rt = Runtime(out)
            inputs = {
                "a": np.asarray(x["a"].values, dtype=np.float64),
                "b": np.asarray(x["b"].values, dtype=np.float64),
            }
            res = rt.compute(inputs, {})

        self.assertIn("predictions", res)
        self.assertEqual(res["predictions"], model.predict(x).tolist())

    def test_package_shape_and_manifest(self):
        import json

        from fnnx.extras.mlflow import package_mlflow_model

        with tempfile.TemporaryDirectory() as tmp:
            model_dir, _, _ = _save_sklearn_model(tmp)
            out = os.path.join(tmp, "m.fnnx")
            package_mlflow_model(model_dir, out, name="sklearn-rf")

            with tarfile.open(out, "r") as tar:
                names = tar.getnames()
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

        # The MLflow dir is embedded verbatim.
        self.assertIn("variant_artifacts/extra_files/mlflow_model/MLmodel", names)

        # Manifest derived from the inferred signature: 2 NDJSON columns.
        self.assertEqual(manifest["variant"], "pyfunc")
        self.assertEqual(manifest["name"], "sklearn-rf")
        # Producer is fnnx.ai with no auto-injected tags.
        self.assertEqual(manifest["producer_name"], "fnnx.ai")
        self.assertEqual(manifest["producer_tags"], [])
        self.assertEqual(len(manifest["inputs"]), 2)
        for spec in manifest["inputs"]:
            self.assertEqual(spec["content_type"], "NDJSON")
            self.assertTrue(spec["dtype"].startswith("Array[float64]"))
        self.assertEqual(manifest["outputs"][0]["name"], "predictions")
        self.assertEqual(manifest["outputs"][0]["dtype"], "ext::mlflow::output")

        # Variant config exposes the wrapper class + fnnx_mlflow cfg.
        self.assertEqual(variant_config["pyfunc_classname"], "MLflowModel")
        cfg = variant_config["extra_values"]["fnnx_mlflow"]
        self.assertEqual(cfg["model_dir"], "mlflow_model")
        self.assertEqual(cfg["input_mode"], "columns")
        self.assertEqual(cfg["column_order"], ["a", "b"])

        # Env pins both mlflow and fnnx via the MLflow requirements + builder.
        deps = [d["package"] for d in env["python3::conda_pip"]["dependencies"]]
        self.assertTrue(any(d.startswith("mlflow") for d in deps))
        self.assertTrue(any(d.startswith("scikit-learn") for d in deps))
        self.assertTrue(any(d.startswith("fnnx") for d in deps))

        # Meta carries the full MLmodel for lossless provenance.
        self.assertEqual(meta[0]["id"], "mlflow-source")
        self.assertEqual(meta[0]["producer"], "fnnx.ai")
        self.assertEqual(meta[0]["payload"]["input_mode"], "columns")
        self.assertIn("mlmodel", meta[0]["payload"])

    def test_extra_pip_dependencies_appended(self):
        import json

        from fnnx.extras.mlflow import package_mlflow_model

        with tempfile.TemporaryDirectory() as tmp:
            model_dir, _, _ = _save_sklearn_model(tmp)
            out = os.path.join(tmp, "m.fnnx")
            package_mlflow_model(
                model_dir, out, extra_pip_dependencies=["my-pkg==1.2.3"]
            )

            with tarfile.open(out, "r") as tar:
                env = json.loads(
                    tar.extractfile("env.json").read().decode()  # type: ignore[union-attr]
                )

        deps = [d["package"] for d in env["python3::conda_pip"]["dependencies"]]
        self.assertIn("my-pkg==1.2.3", deps)

    def test_user_producer_tags_flow_through(self):
        import json

        from fnnx.extras.mlflow import package_mlflow_model

        with tempfile.TemporaryDirectory() as tmp:
            model_dir, _, _ = _save_sklearn_model(tmp)
            out = os.path.join(tmp, "m.fnnx")
            package_mlflow_model(model_dir, out, producer_tags=["team-x", "v2"])

            with tarfile.open(out, "r") as tar:
                manifest = json.loads(
                    tar.extractfile("manifest.json").read().decode()  # type: ignore[union-attr]
                )

        # Exactly the user's tags — no auto-injected mlflow/flavor/version tags.
        self.assertEqual(manifest["producer_tags"], ["team-x", "v2"])


class TestNoMlflowEnv(unittest.TestCase):
    """The local-directory path must not import mlflow at packaging time."""

    def setUp(self) -> None:
        self._saved: dict[str, object] = {}
        # Drop and block any cached mlflow modules.
        for name in list(sys.modules):
            if name == "mlflow" or name.startswith("mlflow."):
                self._saved[name] = sys.modules.pop(name)
        sys.modules["mlflow"] = None  # type: ignore[assignment]

    def tearDown(self) -> None:
        sys.modules.pop("mlflow", None)
        for name, mod in self._saved.items():
            sys.modules[name] = mod  # type: ignore[assignment]

    def _save_model_then_block(self, tmp: str) -> str:
        # Save the model before we block mlflow, then re-block.
        sys.modules.pop("mlflow", None)
        for name, mod in self._saved.items():
            sys.modules[name] = mod  # type: ignore[assignment]
        try:
            model_dir, _, _ = _save_sklearn_model(tmp)
        finally:
            for name in list(sys.modules):
                if name == "mlflow" or name.startswith("mlflow."):
                    self._saved[name] = sys.modules.pop(name)
            sys.modules["mlflow"] = None  # type: ignore[assignment]
        return model_dir

    def test_local_dir_packages_without_mlflow(self):
        from fnnx.extras.mlflow import package_mlflow_model

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._save_model_then_block(tmp)
            out = os.path.join(tmp, "out.fnnx")
            package_mlflow_model(model_dir, out)

            # Resulting package is a valid tarball with the MLflow dir embedded.
            with tarfile.open(out, "r") as tar:
                names = set(tar.getnames())
            self.assertIn("manifest.json", names)
            self.assertIn(
                "variant_artifacts/extra_files/mlflow_model/MLmodel", names
            )

        # mlflow was never imported during packaging.
        self.assertIs(sys.modules.get("mlflow"), None)

    def test_remote_uri_without_mlflow_raises_importerror(self):
        from fnnx.extras.mlflow import package_mlflow_model

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.fnnx")
            with self.assertRaises(ImportError) as cm:
                package_mlflow_model("models:/my_model/3", out)

            self.assertIn("fnnx[mlflow]", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
