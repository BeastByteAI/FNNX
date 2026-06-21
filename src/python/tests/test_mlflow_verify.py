"""Tests for ``verify=True`` smoke testing and ``_example_to_fnnx_inputs``.

The conversion path is unchanged here; we exercise the post-save verification
that loads the FNNX package via ``fnnx.Runtime`` and (when an MLflow input
example exists) runs one ``compute`` to prove the package is portable enough
in the current environment.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pytest


# Unit tests for _example_to_fnnx_inputs do not need mlflow at all.
from fnnx.extras.mlflow import _example_to_fnnx_inputs


# ---------------------------------------------------------------------------
# _example_to_fnnx_inputs unit tests (no mlflow needed)
# ---------------------------------------------------------------------------


class TestExampleToFnnxInputsColumns(unittest.TestCase):
    def test_dataframe_split_by_column_order(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        cfg = {"column_order": ["a", "b"]}
        inputs = _example_to_fnnx_inputs(df, "columns", cfg)

        self.assertEqual(set(inputs.keys()), {"a", "b"})
        np.testing.assert_array_equal(inputs["a"], [1.0, 2.0])
        np.testing.assert_array_equal(inputs["b"], [3.0, 4.0])

    def test_dataframe_without_column_order_uses_df_columns(self):
        df = pd.DataFrame({"x": [10.0], "y": [20.0]})
        inputs = _example_to_fnnx_inputs(df, "columns", {})

        self.assertEqual(set(inputs.keys()), {"x", "y"})
        np.testing.assert_array_equal(inputs["x"], [10.0])

    def test_dict_of_arrays_passes_through(self):
        ex = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
        cfg = {"column_order": ["a", "b"]}
        inputs = _example_to_fnnx_inputs(ex, "columns", cfg)

        self.assertEqual(inputs["a"], [1.0, 2.0])
        self.assertEqual(inputs["b"], [3.0, 4.0])

    def test_list_of_records_reshaped_to_columns(self):
        ex = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        cfg = {"column_order": ["a", "b"]}
        inputs = _example_to_fnnx_inputs(ex, "columns", cfg)

        self.assertEqual(inputs["a"], [1, 3])
        self.assertEqual(inputs["b"], [2, 4])

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            _example_to_fnnx_inputs(42, "columns", {"column_order": ["a"]})


class TestExampleToFnnxInputsTensor(unittest.TestCase):
    def test_single_unnamed_tensor_wrapped_as_input(self):
        arr = np.array([[1.0, 2.0]], dtype=np.float32)
        cfg = {"tensor_names": ["__single__"]}
        inputs = _example_to_fnnx_inputs(arr, "tensor", cfg)

        self.assertEqual(set(inputs.keys()), {"input"})
        self.assertIs(inputs["input"], arr)

    def test_named_tensors_pass_dict_through(self):
        a = np.array([1.0], dtype=np.float32)
        b = np.array([2.0], dtype=np.float32)
        cfg = {"tensor_names": ["a", "b"]}
        inputs = _example_to_fnnx_inputs({"a": a, "b": b}, "tensor", cfg)

        self.assertEqual(set(inputs.keys()), {"a", "b"})
        self.assertIs(inputs["a"], a)
        self.assertIs(inputs["b"], b)

    def test_single_named_tensor_accepts_bare_ndarray(self):
        arr = np.array([1.0, 2.0])
        cfg = {"tensor_names": ["only"]}
        inputs = _example_to_fnnx_inputs(arr, "tensor", cfg)

        self.assertEqual(set(inputs.keys()), {"only"})
        self.assertIs(inputs["only"], arr)

    def test_named_tensors_non_dict_raises(self):
        cfg = {"tensor_names": ["a", "b"]}
        with self.assertRaises(TypeError):
            _example_to_fnnx_inputs([1.0, 2.0], "tensor", cfg)


class TestExampleToFnnxInputsJsonPassthrough(unittest.TestCase):
    def test_json_mode_wraps_under_data(self):
        ex = {"prompt": "hi", "temperature": 0.1}
        inputs = _example_to_fnnx_inputs(ex, "json", {})
        self.assertEqual(inputs, {"data": ex})

    def test_passthrough_mode_wraps_under_data(self):
        inputs = _example_to_fnnx_inputs("hello", "passthrough", {})
        self.assertEqual(inputs, {"data": "hello"})


# ---------------------------------------------------------------------------
# End-to-end verify=True tests (need mlflow + sklearn + pandas)
# ---------------------------------------------------------------------------


pytest.importorskip("mlflow")
pytest.importorskip("sklearn")


def _train_tiny_sklearn():
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
    return model, x, y


def _save_sklearn_model(tmp: str, *, include_input_example: bool = True) -> str:
    import mlflow  # type: ignore[import-not-found]
    from mlflow.models.signature import infer_signature  # type: ignore[import-not-found]

    model, x, _ = _train_tiny_sklearn()
    model_dir = os.path.join(tmp, "mlflow_model")
    sig = infer_signature(x, model.predict(x))
    kwargs = {"signature": sig}
    if include_input_example:
        kwargs["input_example"] = x
    mlflow.sklearn.save_model(model, model_dir, **kwargs)  # type: ignore[attr-defined]
    return model_dir


class TestVerifySuccess(unittest.TestCase):
    def test_verify_true_with_input_example_passes(self):
        from fnnx.extras.mlflow import package_mlflow_model

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _save_sklearn_model(tmp, include_input_example=True)
            out = os.path.join(tmp, "m.fnnx")
            # Should run the input example through Runtime.compute() without
            # raising.
            package_mlflow_model(model_dir, out, verify=True)

            # The package was written (verification did not delete it).
            self.assertTrue(os.path.isfile(out))

    def test_verify_true_without_input_example_passes(self):
        from fnnx.extras.mlflow import package_mlflow_model

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _save_sklearn_model(tmp, include_input_example=False)
            out = os.path.join(tmp, "m.fnnx")
            # No saved input example -> verification only loads the package.
            package_mlflow_model(model_dir, out, verify=True)
            self.assertTrue(os.path.isfile(out))


class TestVerifyLoaderFails(unittest.TestCase):
    def test_load_failure_raises_verification_error(self):
        from fnnx.extras.mlflow import (
            MLflowPackagingVerificationError,
            package_mlflow_model,
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _save_sklearn_model(tmp, include_input_example=False)
            out = os.path.join(tmp, "m.fnnx")

            # Monkeypatch mlflow.pyfunc.load_model to raise so the Runtime warmup
            # blows up, mimicking a non-self-contained model on a clean env.
            import mlflow.pyfunc as pyfunc_mod  # type: ignore[import-not-found]

            def _boom(*_args, **_kwargs):
                raise ImportError("simulated: missing CustomModel class")

            with mock.patch.object(pyfunc_mod, "load_model", _boom):
                with self.assertRaises(MLflowPackagingVerificationError) as cm:
                    package_mlflow_model(model_dir, out, verify=True)

            err = cm.exception
            self.assertIsInstance(err.__cause__, Exception)
            self.assertIn("self-contained", str(err))
            # The package was still written before verification ran.
            self.assertTrue(os.path.isfile(out))


class TestVerifyWithoutMlflow(unittest.TestCase):
    """verify=True must degrade to a skip warning when mlflow is unimportable."""

    def setUp(self) -> None:
        self._saved: dict[str, object] = {}
        # Save and block all mlflow modules to simulate a no-mlflow env.
        for name in list(sys.modules):
            if name == "mlflow" or name.startswith("mlflow."):
                self._saved[name] = sys.modules.pop(name)

    def tearDown(self) -> None:
        sys.modules.pop("mlflow", None)
        for name, mod in self._saved.items():
            sys.modules[name] = mod  # type: ignore[assignment]

    def _save_model_then_block(self, tmp: str) -> str:
        # Re-enable mlflow temporarily to save the source model.
        for name, mod in self._saved.items():
            sys.modules[name] = mod  # type: ignore[assignment]
        try:
            model_dir = _save_sklearn_model(tmp, include_input_example=False)
        finally:
            for name in list(sys.modules):
                if name == "mlflow" or name.startswith("mlflow."):
                    self._saved[name] = sys.modules.pop(name)
            sys.modules["mlflow"] = None  # type: ignore[assignment]
        return model_dir

    def test_verify_true_no_mlflow_warns_and_skips(self):
        from fnnx.extras.mlflow import package_mlflow_model

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = self._save_model_then_block(tmp)
            out = os.path.join(tmp, "m.fnnx")

            # Capture stdout to check the warning.
            with mock.patch(
                "fnnx.extras.mlflow.console.warn"
            ) as warn_mock:
                package_mlflow_model(model_dir, out, verify=True)

            # Package was written.
            self.assertTrue(os.path.isfile(out))
            # Warning was emitted; mlflow stayed un-importable.
            calls = [c.args[0] for c in warn_mock.call_args_list]
            self.assertTrue(
                any("verification skipped" in m for m in calls),
                f"expected skip warning, got {calls!r}",
            )
            # mlflow was never imported during verify.
            self.assertIs(sys.modules.get("mlflow"), None)


if __name__ == "__main__":
    unittest.main()
