"""Unit tests for the static MLflow runtime wrapper and the PyfuncBuilder
``python_version`` override.

These tests deliberately stub out ``mlflow.pyfunc.load_model`` and the loaded
model object so the wrapper's input assembly / param passing / output
normalization logic can be exercised without an actual mlflow install.
"""

from dataclasses import dataclass
import json
import os
import sys
import tarfile
import tempfile
import unittest
from typing import cast
from unittest import mock

import numpy as np
import pandas as pd

from fnnx.variants.pyfunc import Context
from fnnx.extras._mlflow_wrapper import MLflowModel, _to_jsonable
from fnnx.extras.builder import PyfuncBuilder, PyFuncSpec, PYTHON_VERSION


class _FakeContext:
    def __init__(
        self, values: dict, dirs: dict | None = None, device: object | None = None
    ):
        self._values = values
        self._dirs = dirs or {}
        self.device = device

    def get_value(self, key):
        return self._values.get(key)

    def get_dirpath(self, name):
        return self._dirs.get(name)


class _RecordingModel:
    """Stand-in for an mlflow pyfunc model that records calls."""

    def __init__(self, return_value=None):
        self.calls = []
        self.return_value = return_value if return_value is not None else [1, 2, 3]

    def predict(self, payload, params=None):
        self.calls.append({"payload": payload, "params": params})
        return self.return_value


def _make_wrapper(cfg: dict, model: _RecordingModel) -> MLflowModel:
    """Construct an MLflowModel with a fake context and pre-assigned model."""
    ctx = _FakeContext({"fnnx_mlflow": cfg}, dirs={cfg["model_dir"]: "/tmp/fake"})
    wrapper = MLflowModel(cast(Context, ctx))
    wrapper._cfg = cfg
    wrapper.model = model  # type: ignore[assignment]
    return wrapper


class TestComputeColumnsMode(unittest.TestCase):
    def test_dataframe_assembled_from_column_order(self):
        model = _RecordingModel(return_value=np.array([0.1, 0.2]))
        cfg = {
            "model_dir": "mlflow_model",
            "input_mode": "columns",
            "column_order": ["a", "b"],
            "param_names": [],
        }
        wrapper = _make_wrapper(cfg, model)

        inputs = {
            "a": np.array([1.0, 2.0]),
            "b": np.array([10.0, 20.0]),
            # extra columns not in column_order should be ignored
            "c": np.array([99.0, 99.0]),
        }

        out = wrapper.compute(inputs, {})

        self.assertEqual(len(model.calls), 1)
        payload = model.calls[0]["payload"]
        self.assertIsInstance(payload, pd.DataFrame)
        self.assertEqual(list(payload.columns), ["a", "b"])
        np.testing.assert_array_equal(payload["a"].values, [1.0, 2.0])
        np.testing.assert_array_equal(payload["b"].values, [10.0, 20.0])
        self.assertIsNone(model.calls[0]["params"])
        self.assertEqual(out["predictions"], [0.1, 0.2])


class TestComputeTensorMode(unittest.TestCase):
    def test_single_unnamed_tensor_passes_ndarray(self):
        model = _RecordingModel(return_value=np.array([[1.0, 2.0]]))
        cfg = {
            "model_dir": "mlflow_model",
            "input_mode": "tensor",
            "tensor_names": ["__single__"],
            "param_names": [],
        }
        wrapper = _make_wrapper(cfg, model)

        arr = np.array([[1.0, 2.0, 3.0, 4.0]])
        out = wrapper.compute({"input": arr}, {})

        payload = model.calls[0]["payload"]
        self.assertIs(payload, arr)
        self.assertEqual(out["predictions"], [[1.0, 2.0]])

    def test_named_tensors_pass_dict(self):
        model = _RecordingModel(return_value={"y": [0.0]})
        cfg = {
            "model_dir": "mlflow_model",
            "input_mode": "tensor",
            "tensor_names": ["a", "b"],
            "param_names": [],
        }
        wrapper = _make_wrapper(cfg, model)

        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3, 4], dtype=np.int32)
        wrapper.compute({"a": a, "b": b}, {})

        payload = model.calls[0]["payload"]
        self.assertIsInstance(payload, dict)
        self.assertEqual(set(payload.keys()), {"a", "b"})
        self.assertIs(payload["a"], a)
        self.assertIs(payload["b"], b)


class TestComputeJsonAndPassthrough(unittest.TestCase):
    def test_json_mode_unwraps_data(self):
        model = _RecordingModel(return_value={"answer": 42})
        cfg = {
            "model_dir": "mlflow_model",
            "input_mode": "json",
            "param_names": [],
        }
        wrapper = _make_wrapper(cfg, model)

        records = [{"x": 1, "y": "hello"}]
        out = wrapper.compute({"data": records}, {})

        self.assertEqual(model.calls[0]["payload"], records)
        self.assertEqual(out["predictions"], {"answer": 42})

    def test_passthrough_mode_unwraps_data(self):
        model = _RecordingModel(return_value=[1, 2, 3])
        cfg = {
            "model_dir": "mlflow_model",
            "input_mode": "passthrough",
            "param_names": [],
        }
        wrapper = _make_wrapper(cfg, model)

        out = wrapper.compute({"data": "any-payload"}, {})

        self.assertEqual(model.calls[0]["payload"], "any-payload")
        self.assertEqual(out["predictions"], [1, 2, 3])


class TestParamPassing(unittest.TestCase):
    def test_known_params_forwarded_to_predict(self):
        model = _RecordingModel(return_value="ok")
        cfg = {
            "model_dir": "mlflow_model",
            "input_mode": "passthrough",
            "param_names": ["temperature", "max_tokens"],
        }
        wrapper = _make_wrapper(cfg, model)

        wrapper.compute(
            {"data": "x"},
            {"temperature": 0.1, "max_tokens": 256, "unknown": "ignored"},
        )

        params = model.calls[0]["params"]
        self.assertEqual(params, {"temperature": 0.1, "max_tokens": 256})

    def test_missing_params_omitted_no_params_kwarg(self):
        model = _RecordingModel(return_value="ok")
        cfg = {
            "model_dir": "mlflow_model",
            "input_mode": "passthrough",
            "param_names": ["temperature"],
        }
        wrapper = _make_wrapper(cfg, model)

        # No matching dynamic attribute -> predict called WITHOUT params kwarg
        wrapper.compute({"data": "x"}, {})
        self.assertIsNone(model.calls[0]["params"])

    def test_empty_param_names_does_not_pass_params(self):
        model = _RecordingModel(return_value="ok")
        cfg = {
            "model_dir": "mlflow_model",
            "input_mode": "passthrough",
            "param_names": [],
        }
        wrapper = _make_wrapper(cfg, model)

        wrapper.compute({"data": "x"}, {"temperature": 0.5})
        self.assertIsNone(model.calls[0]["params"])


class TestComputeAsyncDelegates(unittest.TestCase):
    def test_async_returns_sync_result(self):
        import asyncio

        model = _RecordingModel(return_value=[1])
        cfg = {
            "model_dir": "mlflow_model",
            "input_mode": "passthrough",
            "param_names": [],
        }
        wrapper = _make_wrapper(cfg, model)

        out = asyncio.run(wrapper.compute_async({"data": "x"}, {}))
        self.assertEqual(out, {"predictions": [1]})


class TestToJsonable(unittest.TestCase):
    def test_primitives_returned_as_is(self):
        self.assertEqual(_to_jsonable(None), None)
        self.assertEqual(_to_jsonable(True), True)
        self.assertEqual(_to_jsonable(7), 7)
        self.assertEqual(_to_jsonable(1.5), 1.5)
        self.assertEqual(_to_jsonable("hello"), "hello")

    def test_numpy_ndarray_and_scalar(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        self.assertEqual(_to_jsonable(arr), [[1, 2], [3, 4]])
        # numpy scalar via .item()
        self.assertEqual(_to_jsonable(np.int64(5)), 5)
        self.assertEqual(_to_jsonable(np.float32(0.5)), 0.5)

    def test_pandas_dataframe_and_series(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        self.assertEqual(
            _to_jsonable(df),
            [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
        )
        s = pd.Series([10, 20, 30])
        self.assertEqual(_to_jsonable(s), [10, 20, 30])

    def test_nested_dict_list_with_numpy(self):
        payload = {
            "vals": np.array([1.0, 2.0]),
            "items": [np.int64(3), {"k": np.float64(1.25)}],
        }
        self.assertEqual(
            _to_jsonable(payload),
            {"vals": [1.0, 2.0], "items": [3, {"k": 1.25}]},
        )

    def test_pydantic_model(self):
        from pydantic import BaseModel

        class M(BaseModel):
            name: str
            value: int

        self.assertEqual(
            _to_jsonable(M(name="a", value=1)),
            {"name": "a", "value": 1},
        )

    def test_dataclass(self):
        @dataclass
        class D:
            x: int
            y: str

        self.assertEqual(_to_jsonable(D(1, "z")), {"x": 1, "y": "z"})

    def test_fallback_str(self):
        class Custom:
            def __str__(self):
                return "custom-repr"

        self.assertEqual(_to_jsonable(Custom()), "custom-repr")


class TestWarmup(unittest.TestCase):
    def test_warmup_invokes_mlflow_pyfunc_load(self):
        """warmup() must call mlflow.pyfunc.load_model on the resolved dir."""

        # Build a fake "mlflow.pyfunc" module that we can inject.
        fake_pyfunc = mock.MagicMock()
        fake_pyfunc.load_model.return_value = "fake-model-object"
        fake_mlflow = mock.MagicMock()
        fake_mlflow.pyfunc = fake_pyfunc

        with mock.patch.dict(
            sys.modules,
            {"mlflow": fake_mlflow, "mlflow.pyfunc": fake_pyfunc},
        ):
            cfg = {
                "model_dir": "mlflow_model",
                "input_mode": "passthrough",
                "param_names": [],
            }
            ctx = _FakeContext(
                {"fnnx_mlflow": cfg},
                dirs={"mlflow_model": "/abs/path/to/mlflow_model"},
            )
            wrapper = MLflowModel(cast(Context, ctx))
            wrapper.warmup()

        fake_pyfunc.load_model.assert_called_once_with("/abs/path/to/mlflow_model")
        self.assertEqual(wrapper.model, "fake-model-object")
        self.assertEqual(wrapper._cfg, cfg)

    def test_warmup_missing_dir_raises(self):
        fake_pyfunc = mock.MagicMock()
        fake_mlflow = mock.MagicMock()
        fake_mlflow.pyfunc = fake_pyfunc

        with mock.patch.dict(
            sys.modules,
            {"mlflow": fake_mlflow, "mlflow.pyfunc": fake_pyfunc},
        ):
            cfg = {
                "model_dir": "mlflow_model",
                "input_mode": "passthrough",
                "param_names": [],
            }
            ctx = _FakeContext({"fnnx_mlflow": cfg}, dirs={})  # not registered
            wrapper = MLflowModel(cast(Context, ctx))
            with self.assertRaises(FileNotFoundError):
                wrapper.warmup()


class TestWarmupDevice(unittest.TestCase):
    """The FNNX device_map should drive mlflow's load-time prediction device."""

    def _warmup_with_device(self, device_attr) -> dict:
        from fnnx.extras._mlflow_wrapper import _MLFLOW_DEVICE_ENV

        seen: dict = {}

        def fake_load(model_dir):
            seen["dir"] = model_dir
            seen["env"] = os.environ.get(_MLFLOW_DEVICE_ENV)
            return "model-obj"

        fake_pyfunc = mock.MagicMock()
        fake_pyfunc.load_model.side_effect = fake_load
        fake_mlflow = mock.MagicMock()
        fake_mlflow.pyfunc = fake_pyfunc

        cfg = {
            "model_dir": "mlflow_model",
            "input_mode": "passthrough",
            "param_names": [],
        }
        ctx = _FakeContext(
            {"fnnx_mlflow": cfg},
            dirs={"mlflow_model": "/abs/x"},
            device=device_attr,  # the wrapper reads Context.device
        )

        with mock.patch.dict(
            sys.modules, {"mlflow": fake_mlflow, "mlflow.pyfunc": fake_pyfunc}
        ):
            wrapper = MLflowModel(cast(Context, ctx))
            wrapper.warmup()
        return seen

    def test_cuda_devicemap_sets_env_during_load_and_restores(self):
        from fnnx.device import DeviceMap
        from fnnx.extras._mlflow_wrapper import _MLFLOW_DEVICE_ENV

        self.assertNotIn(_MLFLOW_DEVICE_ENV, os.environ)
        seen = self._warmup_with_device(
            DeviceMap(accelerator="cuda", node_device_map={})
        )
        self.assertEqual(seen["env"], "cuda")
        # restored after load — process env not leaked
        self.assertNotIn(_MLFLOW_DEVICE_ENV, os.environ)

    def test_cpu_string_forces_cpu(self):
        seen = self._warmup_with_device("cpu")
        self.assertEqual(seen["env"], "cpu")

    def test_cuda_indexed_preserved(self):
        from fnnx.device import DeviceMap

        seen = self._warmup_with_device(
            DeviceMap(accelerator="cuda:1", node_device_map={})
        )
        self.assertEqual(seen["env"], "cuda:1")

    def test_unknown_accelerator_leaves_env_unset(self):
        seen = self._warmup_with_device("auto")
        self.assertIsNone(seen["env"])

    def test_external_env_var_wins(self):
        from fnnx.device import DeviceMap
        from fnnx.extras._mlflow_wrapper import _MLFLOW_DEVICE_ENV

        os.environ[_MLFLOW_DEVICE_ENV] = "cuda:3"
        try:
            seen = self._warmup_with_device(
                DeviceMap(accelerator="cpu", node_device_map={})
            )
            self.assertEqual(seen["env"], "cuda:3")  # device_map did not override
        finally:
            os.environ.pop(_MLFLOW_DEVICE_ENV, None)


class TestBuilderPythonVersion(unittest.TestCase):
    def _build(self, python_version: str | None) -> dict:
        # Use the static wrapper module as the PyFunc source — its filepath
        # exists on disk so PyfuncBuilder can read it without a real PyFunc class.
        from fnnx.extras import _mlflow_wrapper

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "test.fnnx")
            kwargs = {}
            if python_version is not None:
                kwargs["python_version"] = python_version
            builder = PyfuncBuilder(
                pyfunc=PyFuncSpec(
                    filepath=_mlflow_wrapper.__file__, class_name="MLflowModel"
                ),
                **kwargs,
            )
            builder.add_runtime_dependency("mlflow==3.4.0")
            builder.save(out_path)

            with tarfile.open(out_path, "r") as tar:
                env_member = tar.extractfile("env.json")
                assert env_member is not None
                env = json.loads(env_member.read())
        return env

    def test_python_version_override_used(self):
        env = self._build(python_version="3.11.4")
        self.assertEqual(env["python3::conda_pip"]["python_version"], "3.11.4")

    def test_default_python_version_when_unset(self):
        env = self._build(python_version=None)
        self.assertEqual(
            env["python3::conda_pip"]["python_version"], PYTHON_VERSION
        )


if __name__ == "__main__":
    unittest.main()
