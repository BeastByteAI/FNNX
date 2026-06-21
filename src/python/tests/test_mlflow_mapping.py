"""Unit tests for the MLflow signature / env / meta / warning helpers.

The mapping helpers (`_map_input_schema`, `_map_params`, `_map_env`,
`_self_containment_warnings`, `_build_meta_payload`, `create_meta_callback`)
consume the same normalized JSON entries that MLflow serializes into the
`MLmodel` file, so the bulk of these tests build those entries directly and
do not require an mlflow install. The reader equivalence test gates on
mlflow via `pytest.importorskip`.
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import unittest
from textwrap import dedent

import pytest

from fnnx.extras.builder import File
from fnnx.extras.pydantic_models.manifest import NDJSON, JSON, Var
from fnnx.extras.mlflow import (
    MLModelInfo,
    _build_meta_payload,
    _load_model_info_yaml,
    _map_env,
    _map_input_schema,
    _map_output_schema,
    _map_params,
    _parse_requirements_txt,
    _parse_signature_field,
    _self_containment_warnings,
    create_meta_callback,
)


# ---------------------------------------------------------------------------
# _map_input_schema
# ---------------------------------------------------------------------------


class TestMapInputSchemaPassthrough(unittest.TestCase):
    def test_no_signature_yields_passthrough(self):
        specs, mode, cfg, warnings = _map_input_schema(None)

        self.assertEqual(mode, "passthrough")
        self.assertEqual(cfg, {"input_mode": "passthrough"})
        self.assertEqual(len(specs), 1)
        spec = specs[0]
        self.assertIsInstance(spec, JSON)
        self.assertEqual(spec.name, "data")
        self.assertEqual(spec.dtype, "ext::mlflow::input")
        self.assertTrue(warnings, "expected a warning about missing signature")
        self.assertIn("passthrough", warnings[0])


class TestMapInputSchemaColumns(unittest.TestCase):
    def test_named_double_columns(self):
        entries = [
            {"type": "double", "name": "sepal_len"},
            {"type": "double", "name": "sepal_wid"},
            {"type": "double", "name": "petal_len"},
            {"type": "double", "name": "petal_wid"},
        ]
        specs, mode, cfg, warnings = _map_input_schema(entries)

        self.assertEqual(mode, "columns")
        self.assertEqual(warnings, [])
        self.assertEqual(
            cfg,
            {
                "input_mode": "columns",
                "column_order": [
                    "sepal_len",
                    "sepal_wid",
                    "petal_len",
                    "petal_wid",
                ],
            },
        )
        self.assertEqual(len(specs), 4)
        for spec in specs:
            assert isinstance(spec, NDJSON)
            self.assertEqual(spec.dtype, "Array[float64]")
            self.assertEqual(spec.shape, [-1])

    def test_columns_unnamed_get_col_index_names(self):
        entries = [{"type": "long"}, {"type": "long"}]
        specs, mode, cfg, _ = _map_input_schema(entries)

        self.assertEqual(mode, "columns")
        self.assertEqual(cfg["column_order"], ["col_0", "col_1"])
        for spec in specs:
            assert isinstance(spec, NDJSON)
            self.assertEqual(spec.dtype, "Array[int64]")

    def test_columns_dtype_table_covers_all_simple_scalars(self):
        cases = {
            "boolean": "bool",
            "integer": "int32",
            "long": "int64",
            "float": "float32",
            "double": "float64",
            "string": "string",
        }
        for mlflow_type, token in cases.items():
            entries = [{"type": mlflow_type, "name": "x"}]
            specs, mode, _, _ = _map_input_schema(entries)
            assert mode == "columns"
            assert isinstance(specs[0], NDJSON)
            self.assertEqual(
                specs[0].dtype,
                f"Array[{token}]",
                msg=f"{mlflow_type} should map to Array[{token}]",
            )


class TestMapInputSchemaTensor(unittest.TestCase):
    def test_single_unnamed_tensor(self):
        entries = [
            {
                "type": "tensor",
                "tensor-spec": {"dtype": "float32", "shape": [-1, 4]},
            }
        ]
        specs, mode, cfg, warnings = _map_input_schema(entries)

        self.assertEqual(mode, "tensor")
        self.assertEqual(warnings, [])
        self.assertEqual(cfg["input_mode"], "tensor")
        self.assertEqual(cfg["tensor_names"], ["__single__"])
        self.assertEqual(len(specs), 1)
        assert isinstance(specs[0], NDJSON)
        self.assertEqual(specs[0].name, "input")
        self.assertEqual(specs[0].dtype, "Array[float32]")
        self.assertEqual(specs[0].shape, [-1, 4])

    def test_multiple_named_tensors(self):
        entries = [
            {
                "type": "tensor",
                "name": "a",
                "tensor-spec": {"dtype": "float32", "shape": [None, 3]},
            },
            {
                "type": "tensor",
                "name": "b",
                "tensor-spec": {"dtype": "int64", "shape": [-1]},
            },
        ]
        specs, mode, cfg, _ = _map_input_schema(entries)

        self.assertEqual(mode, "tensor")
        self.assertEqual(cfg["tensor_names"], ["a", "b"])
        self.assertEqual(len(specs), 2)
        names = [s.name for s in specs]
        self.assertEqual(names, ["a", "b"])
        assert isinstance(specs[0], NDJSON)
        self.assertEqual(specs[0].dtype, "Array[float32]")
        self.assertEqual(specs[0].shape, [-1, 3])
        assert isinstance(specs[1], NDJSON)
        self.assertEqual(specs[1].dtype, "Array[int64]")
        self.assertEqual(specs[1].shape, [-1])


class TestMapInputSchemaJson(unittest.TestCase):
    def test_nested_object_forces_json(self):
        entries = [
            {
                "name": "messages",
                "type": "array",
                "items": {"type": "object", "properties": {}},
            }
        ]
        specs, mode, cfg, _ = _map_input_schema(entries)

        self.assertEqual(mode, "json")
        self.assertEqual(cfg, {"input_mode": "json"})
        self.assertEqual(len(specs), 1)
        self.assertIsInstance(specs[0], JSON)
        self.assertEqual(specs[0].name, "data")
        self.assertEqual(specs[0].dtype, "ext::mlflow::input")

    def test_datetime_column_forces_json(self):
        entries = [
            {"type": "double", "name": "y"},
            {"type": "datetime", "name": "ds"},
        ]
        _, mode, _, _ = _map_input_schema(entries)
        self.assertEqual(mode, "json")

    def test_binary_column_forces_json(self):
        entries = [
            {"type": "double", "name": "x"},
            {"type": "binary", "name": "blob"},
        ]
        _, mode, _, _ = _map_input_schema(entries)
        self.assertEqual(mode, "json")

    def test_mixed_tensor_and_scalar_is_not_tensor_mode(self):
        entries = [
            {
                "type": "tensor",
                "tensor-spec": {"dtype": "float32", "shape": [-1]},
            },
            {"type": "double", "name": "y"},
        ]
        # Mixed tensor + scalar: scalar entries cannot be treated as
        # tensor; loop falls through to json (the most permissive option).
        _, mode, _, _ = _map_input_schema(entries)
        self.assertEqual(mode, "json")


class TestMapInputSchemaEmpty(unittest.TestCase):
    def test_empty_signature_inputs_list_raises(self):
        with self.assertRaises(ValueError):
            _map_input_schema([])


# ---------------------------------------------------------------------------
# _map_output_schema
# ---------------------------------------------------------------------------


class TestMapOutputSchema(unittest.TestCase):
    def test_outputs_always_single_permissive_predictions(self):
        specs, warnings = _map_output_schema()
        self.assertEqual(warnings, [])
        self.assertEqual(len(specs), 1)
        out = specs[0]
        self.assertIsInstance(out, JSON)
        self.assertEqual(out.name, "predictions")
        self.assertEqual(out.dtype, "ext::mlflow::output")


# ---------------------------------------------------------------------------
# _map_params
# ---------------------------------------------------------------------------


class TestMapParams(unittest.TestCase):
    def test_none_yields_empty(self):
        vars_, names = _map_params(None)
        self.assertEqual(vars_, [])
        self.assertEqual(names, [])

    def test_empty_yields_empty(self):
        vars_, names = _map_params([])
        self.assertEqual(vars_, [])
        self.assertEqual(names, [])

    def test_params_become_dynamic_attributes(self):
        entries = [
            {"name": "temperature", "type": "double", "default": 0.7, "shape": None},
            {"name": "max_tokens", "type": "long", "default": 256, "shape": None},
        ]
        vars_, names = _map_params(entries)
        self.assertEqual(names, ["temperature", "max_tokens"])
        self.assertEqual(len(vars_), 2)
        for v in vars_:
            self.assertIsInstance(v, Var)
        self.assertIn("temperature", vars_[0].description)
        self.assertIn("type=double", vars_[0].description)
        self.assertIn("default=0.7", vars_[0].description)

    def test_param_without_name_raises(self):
        with self.assertRaises(ValueError):
            _map_params([{"type": "double", "default": 0.0}])


# ---------------------------------------------------------------------------
# _parse_signature_field
# ---------------------------------------------------------------------------


class TestParseSignatureField(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_parse_signature_field(None))

    def test_json_string_parsed(self):
        s = '[{"type": "double", "name": "x"}]'
        out = _parse_signature_field(s)
        self.assertEqual(out, [{"type": "double", "name": "x"}])

    def test_already_decoded_list_passthrough(self):
        v = [{"type": "long", "name": "k"}]
        self.assertEqual(_parse_signature_field(v), v)

    def test_non_list_raises(self):
        with self.assertRaises(ValueError):
            _parse_signature_field('{"type": "double"}')


# ---------------------------------------------------------------------------
# _map_env
# ---------------------------------------------------------------------------


class TestMapEnv(unittest.TestCase):
    def _write(self, tmp: str, name: str, body: str) -> str:
        path = os.path.join(tmp, name)
        with open(path, "w") as f:
            f.write(body)
        return path

    def test_reads_python_env_and_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(
                tmp,
                "python_env.yaml",
                dedent(
                    """
                    python: 3.9.8
                    build_dependencies:
                      - pip==21.3.1
                      - setuptools==58.0.4
                      - wheel==0.37.0
                    dependencies:
                      - -r requirements.txt
                    """
                ).strip(),
            )
            self._write(
                tmp,
                "requirements.txt",
                dedent(
                    """
                    # auto-generated by mlflow
                    mlflow==3.4.0
                    scikit-learn==1.4.0
                    cloudpickle==3.0.0
                    -r extra.txt
                    --index-url https://example.com/simple
                    """
                ).strip(),
            )

            py, build, runtime = _map_env(tmp, flavors=None)

            self.assertEqual(py, "3.9.8")
            self.assertEqual(
                build,
                ["pip==21.3.1", "setuptools==58.0.4", "wheel==0.37.0"],
            )
            self.assertEqual(
                runtime,
                ["mlflow==3.4.0", "scikit-learn==1.4.0", "cloudpickle==3.0.0"],
            )

    def test_falls_back_to_pyfunc_python_version_without_env_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "requirements.txt", "mlflow==3.4.0\n")
            flavors = {"python_function": {"python_version": "3.11.4"}}
            py, build, runtime = _map_env(tmp, flavors=flavors)
            self.assertEqual(py, "3.11.4")
            self.assertEqual(build, [])
            self.assertEqual(runtime, ["mlflow==3.4.0"])

    def test_no_env_yaml_no_flavor_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            py, build, runtime = _map_env(tmp, flavors={})
            self.assertIsNone(py)
            self.assertEqual(build, [])
            self.assertEqual(runtime, [])


class TestParseRequirementsTxt(unittest.TestCase):
    def test_skips_comments_blank_and_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.txt")
            with open(path, "w") as f:
                f.write(
                    "\n# comment\nmlflow==3.4.0\n  \n"
                    "-r other.txt\n-c constraints.txt\n--pre\n"
                    "scikit-learn==1.4.0\n"
                )
            deps = _parse_requirements_txt(path)
            self.assertEqual(deps, ["mlflow==3.4.0", "scikit-learn==1.4.0"])


# ---------------------------------------------------------------------------
# _self_containment_warnings
# ---------------------------------------------------------------------------


def _info_with_flavors(flavors: dict) -> MLModelInfo:
    return MLModelInfo(
        flavors=flavors,
        signature_inputs=None,
        signature_outputs=None,
        signature_params=None,
        metadata=None,
        mlflow_version=None,
        saved_input_example_info=None,
        raw_mlmodel={},
    )


class TestSelfContainmentWarnings(unittest.TestCase):
    def test_no_warning_when_code_dir_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "code"))
            info = _info_with_flavors(
                {"python_function": {"loader_module": "mlflow.sklearn"}}
            )
            self.assertEqual(_self_containment_warnings(info, tmp), [])

    def test_no_warning_when_models_from_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = _info_with_flavors(
                {
                    "python_function": {
                        "loader_module": "mlflow.langchain",
                        "model_code_path": "langchain_model.py",
                    }
                }
            )
            self.assertEqual(_self_containment_warnings(info, tmp), [])

    def test_no_warning_for_native_flavors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for loader in ("mlflow.onnx", "mlflow.xgboost", "mlflow.lightgbm"):
                info = _info_with_flavors(
                    {"python_function": {"loader_module": loader}}
                )
                self.assertEqual(
                    _self_containment_warnings(info, tmp), [],
                    msg=f"{loader} should emit no warning",
                )

    def test_pickle_loader_without_code_dir_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            for loader in ("mlflow.pytorch", "mlflow.sklearn", "mlflow.pyfunc.model"):
                info = _info_with_flavors(
                    {"python_function": {"loader_module": loader}}
                )
                warnings = _self_containment_warnings(info, tmp)
                self.assertTrue(
                    warnings, msg=f"{loader} should emit a warning"
                )
                self.assertIn(loader, warnings[0])
                self.assertIn("code_paths", warnings[0])

    def test_transformers_save_pretrained_false_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = _info_with_flavors(
                {
                    "python_function": {"loader_module": "mlflow.transformers"},
                    "transformers": {"save_pretrained": False},
                }
            )
            warnings = _self_containment_warnings(info, tmp)
            self.assertEqual(len(warnings), 1)
            self.assertIn("Hugging Face Hub", warnings[0])

    def test_transformers_save_pretrained_true_no_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = _info_with_flavors(
                {
                    "python_function": {"loader_module": "mlflow.transformers"},
                    "transformers": {"save_pretrained": True},
                }
            )
            self.assertEqual(_self_containment_warnings(info, tmp), [])


# ---------------------------------------------------------------------------
# _build_meta_payload + create_meta_callback
# ---------------------------------------------------------------------------


class TestMetaPayload(unittest.TestCase):
    def _sample_info(self) -> MLModelInfo:
        return MLModelInfo(
            flavors={
                "python_function": {"loader_module": "mlflow.sklearn"},
                "sklearn": {"sklearn_version": "1.4.0"},
            },
            signature_inputs=[{"type": "double", "name": "x"}],
            signature_outputs=[{"type": "double"}],
            signature_params=None,
            metadata={"foo": "bar"},
            mlflow_version="3.4.0",
            saved_input_example_info=None,
            raw_mlmodel={"flavors": {"sklearn": {}}, "mlflow_version": "3.4.0"},
        )

    def test_payload_contains_provenance_fields(self):
        info = self._sample_info()
        payload = _build_meta_payload(info, input_mode="columns")
        self.assertEqual(payload["loader_module"], "mlflow.sklearn")
        self.assertEqual(payload["mlflow_version"], "3.4.0")
        self.assertEqual(payload["input_mode"], "columns")
        self.assertEqual(
            set(payload["flavors"]), {"python_function", "sklearn"}
        )
        self.assertEqual(payload["mlmodel"], info.raw_mlmodel)
        self.assertEqual(
            payload["signature"]["inputs"], info.signature_inputs
        )

    def test_create_meta_callback_writes_one_entry(self):
        info = self._sample_info()
        cb = create_meta_callback(info, "columns", producer_version="0.0.99")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.tar")
            f = File(path)
            cb(f)
            f.close()

            with tarfile.open(path, "r") as tar:
                member = tar.getmember("meta.json")
                stream = tar.extractfile(member)
                assert stream is not None
                entries = json.loads(stream.read().decode())

        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["id"], "mlflow-source")
        self.assertEqual(entry["producer"], "fnnx.extras.mlflow")
        self.assertEqual(entry["producer_version"], "0.0.99")
        self.assertIn("mlflow", entry["producer_tags"])
        self.assertIn("sklearn", entry["producer_tags"])
        self.assertIn("mlflow==3.4.0", entry["producer_tags"])
        self.assertEqual(entry["payload"]["input_mode"], "columns")
        self.assertEqual(entry["payload"]["loader_module"], "mlflow.sklearn")


# ---------------------------------------------------------------------------
# _load_model_info_yaml
# ---------------------------------------------------------------------------


_SAMPLE_MLMODEL = dedent(
    """
    artifact_path: model
    flavors:
      python_function:
        env: conda.yaml
        loader_module: mlflow.sklearn
        python_version: 3.10.4
      sklearn:
        pickled_model: model.pkl
        sklearn_version: 1.4.0
    mlflow_version: 3.4.0
    metadata:
      task: classification
    saved_input_example_info:
      artifact_path: input_example.json
      type: dataframe
    signature:
      inputs: '[{"type": "double", "name": "x"}, {"type": "double", "name": "y"}]'
      outputs: '[{"type": "long"}]'
      params: '[{"name": "temperature", "type": "double", "default": 0.7, "shape": null}]'
    """
).strip()


class TestLoadModelInfoYaml(unittest.TestCase):
    def test_parses_hand_authored_mlmodel(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "MLmodel"), "w") as f:
                f.write(_SAMPLE_MLMODEL)
            info = _load_model_info_yaml(tmp)

        self.assertEqual(info.mlflow_version, "3.4.0")
        self.assertEqual(set(info.flavors.keys()), {"python_function", "sklearn"})
        self.assertEqual(info.loader_module, "mlflow.sklearn")
        self.assertEqual(info.primary_flavor, "sklearn")
        self.assertEqual(info.metadata, {"task": "classification"})
        self.assertEqual(
            info.signature_inputs,
            [
                {"type": "double", "name": "x"},
                {"type": "double", "name": "y"},
            ],
        )
        self.assertEqual(info.signature_outputs, [{"type": "long"}])
        assert info.signature_params is not None
        self.assertEqual(info.signature_params[0]["name"], "temperature")
        self.assertEqual(
            info.saved_input_example_info,
            {"artifact_path": "input_example.json", "type": "dataframe"},
        )

    def test_missing_mlmodel_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                _load_model_info_yaml(tmp)

    def test_missing_signature_is_none(self):
        body = dedent(
            """
            flavors:
              python_function:
                loader_module: mlflow.pyfunc.model
            mlflow_version: 3.4.0
            """
        ).strip()
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "MLmodel"), "w") as f:
                f.write(body)
            info = _load_model_info_yaml(tmp)
        self.assertIsNone(info.signature_inputs)
        self.assertIsNone(info.signature_outputs)
        self.assertIsNone(info.signature_params)


# ---------------------------------------------------------------------------
# Reader equivalence (mlflow vs YAML) — only runs with mlflow installed.
# ---------------------------------------------------------------------------


class TestReaderEquivalence(unittest.TestCase):
    def test_mlflow_and_yaml_readers_agree(self):
        mlflow = pytest.importorskip("mlflow")
        pytest.importorskip("sklearn")
        pytest.importorskip("pandas")

        from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-not-found]
        import pandas as pd
        from mlflow.models.signature import infer_signature  # type: ignore[import-not-found]

        from fnnx.extras.mlflow import (
            _load_model_info_mlflow,
            _load_model_info_yaml,
        )

        # Tiny deterministic model + signature.
        x = pd.DataFrame(
            {"a": [0.0, 1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0, 0.0]}
        )
        y = [0, 1, 0, 1]
        model = RandomForestClassifier(n_estimators=2, random_state=0)
        model.fit(x, y)
        sig = infer_signature(x, model.predict(x))

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "m")
            mlflow.sklearn.save_model(  # type: ignore[attr-defined]
                model, model_dir, signature=sig
            )
            info_mlflow = _load_model_info_mlflow(model_dir)
            info_yaml = _load_model_info_yaml(model_dir)

        self.assertEqual(info_mlflow.signature_inputs, info_yaml.signature_inputs)
        self.assertEqual(
            info_mlflow.signature_outputs, info_yaml.signature_outputs
        )
        self.assertEqual(info_mlflow.signature_params, info_yaml.signature_params)
        self.assertEqual(info_mlflow.flavors, info_yaml.flavors)
        self.assertEqual(info_mlflow.mlflow_version, info_yaml.mlflow_version)


if __name__ == "__main__":
    unittest.main()
