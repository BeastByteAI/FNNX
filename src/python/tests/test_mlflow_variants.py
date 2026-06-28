"""Broader end-to-end coverage for ``package_mlflow_model``.

Each test trains/saves a tiny deterministic MLflow model exercising a specific
input mode (json/custom dict output, tensor, passthrough, params), packages it
with the converter, loads via ``fnnx.Runtime``, and asserts a round-trip
behavior matching direct invocation through MLflow.

Models-from-code is used for the custom-PythonModel scenarios (json / params)
because the class must be loadable in a fresh process at warmup time without
relying on test-module import paths.
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import textwrap
import unittest
from unittest import mock

import pytest


pytest.importorskip("mlflow")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Custom PythonModel (json input_mode, dict output)
# ---------------------------------------------------------------------------


_DICT_MODEL_SCRIPT = textwrap.dedent(
    """
    import mlflow


    class DictModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            # model_input arrives as a list of records (pyfunc serialization).
            if hasattr(model_input, "to_dict"):
                records = model_input.to_dict(orient="records")
            else:
                records = model_input
            first = records[0] if records else {}
            items = first.get("items", []) if isinstance(first, dict) else []
            return {"count": len(items), "items": list(items)}


    mlflow.models.set_model(DictModel())
    """
).lstrip()


def _save_dict_model(tmp: str) -> str:
    """Save a custom PythonModel (models-from-code) with a nested ColSpec signature."""
    import mlflow  # type: ignore[import-not-found]
    from mlflow.models import ModelSignature  # type: ignore[import-not-found]
    from mlflow.types import DataType  # type: ignore[import-not-found]
    from mlflow.types.schema import Array, ColSpec, Schema  # type: ignore[import-not-found]

    script_path = os.path.join(tmp, "dict_model_script.py")
    with open(script_path, "w") as f:
        f.write(_DICT_MODEL_SCRIPT)

    signature = ModelSignature(
        inputs=Schema([ColSpec(name="items", type=Array(DataType.string))]),
        outputs=Schema([ColSpec(type=DataType.string)]),
    )

    model_dir = os.path.join(tmp, "dict_model")
    mlflow.pyfunc.save_model(  # type: ignore[attr-defined]
        path=model_dir,
        python_model=script_path,
        signature=signature,
    )
    return model_dir


class TestCustomPythonModelJsonMode(unittest.TestCase):
    """Nested-ColSpec signature → json input_mode; dict output is normalized."""

    def test_round_trip(self):
        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _save_dict_model(tmp)
            out = os.path.join(tmp, "dict.fnnx")
            package_mlflow_model(model_dir, out)

            with tarfile.open(out, "r") as tar:
                variant_config = json.loads(
                    tar.extractfile("variant_config.json").read().decode()  # type: ignore[union-attr]
                )
                manifest = json.loads(
                    tar.extractfile("manifest.json").read().decode()  # type: ignore[union-attr]
                )

            cfg = variant_config["extra_values"]["fnnx_mlflow"]
            self.assertEqual(cfg["input_mode"], "json")
            self.assertEqual(len(manifest["inputs"]), 1)
            self.assertEqual(manifest["inputs"][0]["name"], "data")
            self.assertEqual(manifest["inputs"][0]["content_type"], "JSON")
            self.assertEqual(manifest["inputs"][0]["dtype"], "ext::mlflow::input")

            rt = Runtime(out)
            records = [{"items": ["a", "b", "c"]}]
            result = rt.compute({"data": records}, {})

        self.assertIn("predictions", result)
        predictions = result["predictions"]
        self.assertEqual(predictions, {"count": 3, "items": ["a", "b", "c"]})


# ---------------------------------------------------------------------------
# TensorSpec signature → tensor input_mode
# ---------------------------------------------------------------------------


def _save_tensor_model(tmp: str):
    """Train a tiny regressor and save it with a single unnamed TensorSpec signature."""
    import mlflow  # type: ignore[import-not-found]
    import numpy as np
    from mlflow.models import ModelSignature  # type: ignore[import-not-found]
    from mlflow.types.schema import Schema, TensorSpec  # type: ignore[import-not-found]
    from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-not-found]

    rng = np.random.default_rng(0)
    X = rng.random((20, 4)).astype(np.float32)
    y = rng.random(20).astype(np.float64)
    model = RandomForestRegressor(n_estimators=3, random_state=0)
    model.fit(X, y)

    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), [-1, 4])]),
        outputs=Schema([TensorSpec(np.dtype("float64"), [-1])]),
    )

    model_dir = os.path.join(tmp, "tensor_model")
    mlflow.sklearn.save_model(model, model_dir, signature=signature)  # type: ignore[attr-defined]
    return model_dir, model, X


class TestTensorMode(unittest.TestCase):
    def test_unnamed_tensor_round_trip(self):
        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            model_dir, model, X = _save_tensor_model(tmp)
            out = os.path.join(tmp, "tensor.fnnx")
            package_mlflow_model(model_dir, out)

            with tarfile.open(out, "r") as tar:
                variant_config = json.loads(
                    tar.extractfile("variant_config.json").read().decode()  # type: ignore[union-attr]
                )
                manifest = json.loads(
                    tar.extractfile("manifest.json").read().decode()  # type: ignore[union-attr]
                )

            cfg = variant_config["extra_values"]["fnnx_mlflow"]
            self.assertEqual(cfg["input_mode"], "tensor")
            self.assertEqual(cfg["tensor_names"], ["__single__"])
            self.assertEqual(len(manifest["inputs"]), 1)
            spec = manifest["inputs"][0]
            self.assertEqual(spec["name"], "input")
            self.assertEqual(spec["content_type"], "NDJSON")
            self.assertEqual(spec["dtype"], "Array[float32]")
            self.assertEqual(spec["shape"], ["batch", 4])

            rt = Runtime(out)
            sample = X[:5]
            result = rt.compute({"input": sample}, {})

        self.assertIn("predictions", result)
        expected = model.predict(sample).tolist()
        self.assertEqual(result["predictions"], expected)


# ---------------------------------------------------------------------------
# No-signature model → passthrough input_mode (with warning)
# ---------------------------------------------------------------------------


def _save_no_signature_model(tmp: str):
    import mlflow  # type: ignore[import-not-found]
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-not-found]

    x = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [3.0, 2.0, 1.0, 0.0],
        }
    )
    y = [0, 1, 0, 1]
    model = RandomForestClassifier(n_estimators=3, random_state=0)
    model.fit(x, y)

    model_dir = os.path.join(tmp, "no_sig_model")
    mlflow.sklearn.save_model(model, model_dir)  # type: ignore[attr-defined]
    return model_dir, model, x


class TestPassthroughMode(unittest.TestCase):
    def test_round_trip_and_warning(self):
        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            model_dir, model, x = _save_no_signature_model(tmp)
            out = os.path.join(tmp, "passthrough.fnnx")

            with mock.patch("fnnx.extras.mlflow.console.warn") as warn_mock:
                package_mlflow_model(model_dir, out)

            with tarfile.open(out, "r") as tar:
                variant_config = json.loads(
                    tar.extractfile("variant_config.json").read().decode()  # type: ignore[union-attr]
                )
                manifest = json.loads(
                    tar.extractfile("manifest.json").read().decode()  # type: ignore[union-attr]
                )

            cfg = variant_config["extra_values"]["fnnx_mlflow"]
            self.assertEqual(cfg["input_mode"], "passthrough")
            self.assertEqual(len(manifest["inputs"]), 1)
            self.assertEqual(manifest["inputs"][0]["name"], "data")
            self.assertEqual(manifest["inputs"][0]["dtype"], "ext::mlflow::input")

            warned = [c.args[0] for c in warn_mock.call_args_list]
            self.assertTrue(
                any("passthrough" in m for m in warned),
                f"expected passthrough warning, got {warned!r}",
            )

            rt = Runtime(out)
            # In passthrough mode the wrapper hands inputs["data"] straight to
            # predict(); pyfunc accepts a DataFrame for sklearn models.
            result = rt.compute({"data": x}, {})

        self.assertIn("predictions", result)
        self.assertEqual(result["predictions"], model.predict(x).tolist())


# ---------------------------------------------------------------------------
# Params schema → dynamic_attributes + predict(params=...)
# ---------------------------------------------------------------------------


_PARAMS_MODEL_SCRIPT = textwrap.dedent(
    """
    import mlflow


    class ParamsModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            params = params or {}
            return {
                "temperature_used": params.get("temperature", -1.0),
                "max_tokens_used": params.get("max_tokens", -1),
            }


    mlflow.models.set_model(ParamsModel())
    """
).lstrip()


def _save_params_model(tmp: str) -> str:
    import mlflow  # type: ignore[import-not-found]
    from mlflow.models import ModelSignature  # type: ignore[import-not-found]
    from mlflow.types import DataType  # type: ignore[import-not-found]
    from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema  # type: ignore[import-not-found]

    script_path = os.path.join(tmp, "params_model_script.py")
    with open(script_path, "w") as f:
        f.write(_PARAMS_MODEL_SCRIPT)

    signature = ModelSignature(
        inputs=Schema([ColSpec(name="prompt", type=DataType.string)]),
        outputs=Schema([ColSpec(type=DataType.string)]),
        params=ParamSchema(
            [
                ParamSpec("temperature", DataType.double, 0.7),
                ParamSpec("max_tokens", DataType.long, 256),
            ]
        ),
    )

    model_dir = os.path.join(tmp, "params_model")
    mlflow.pyfunc.save_model(  # type: ignore[attr-defined]
        path=model_dir,
        python_model=script_path,
        signature=signature,
    )
    return model_dir


class TestParamsModeDynamicAttributes(unittest.TestCase):
    """Params schema → dynamic_attributes; values reach predict(..., params=...)."""

    def test_manifest_and_round_trip(self):
        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _save_params_model(tmp)
            out = os.path.join(tmp, "params.fnnx")
            package_mlflow_model(model_dir, out)

            with tarfile.open(out, "r") as tar:
                manifest = json.loads(
                    tar.extractfile("manifest.json").read().decode()  # type: ignore[union-attr]
                )
                variant_config = json.loads(
                    tar.extractfile("variant_config.json").read().decode()  # type: ignore[union-attr]
                )

            attrs = manifest["dynamic_attributes"]
            attr_names = {a["name"] for a in attrs}
            self.assertEqual(attr_names, {"temperature", "max_tokens"})

            cfg = variant_config["extra_values"]["fnnx_mlflow"]
            self.assertEqual(sorted(cfg["param_names"]), ["max_tokens", "temperature"])

            rt = Runtime(out)
            # Pass temperature, omit max_tokens (lets MLflow default apply).
            result = rt.compute(
                {"prompt": ["hello"]},
                {"temperature": 0.1, "max_tokens": 42},
            )

        self.assertIn("predictions", result)
        predictions = result["predictions"]
        self.assertEqual(predictions["temperature_used"], 0.1)
        self.assertEqual(predictions["max_tokens_used"], 42)


if __name__ == "__main__":
    unittest.main()
