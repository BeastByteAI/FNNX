"""End-to-end tests for the MLflow → FNNX converter on LangChain/LangGraph
models-from-code (offline).

Each test logs a deterministic models-from-code script via
``mlflow.langchain.save_model``, packages with ``package_mlflow_model``,
loads via ``fnnx.Runtime``, and asserts:

* the embedded ``mlflow_model/`` contains the model-code ``.py`` (the file
  models-from-code points ``model_code_path`` at),
* **no** code-availability self-containment warning fires (models-from-code
  triggers the self-contained-ish path even for the ``mlflow.langchain``
  loader),
* a round-trip ``compute`` on the converted package matches what the
  embedded runnable/graph returns when invoked directly via
  ``mlflow.pyfunc.load_model``.

Both fixtures use an ``Object`` ColSpec so the converter selects
``input_mode="json"`` — the natural shape for GenAI-style I/O.
"""

from __future__ import annotations

import os
import tarfile
import tempfile
import unittest
from unittest import mock

import pytest


pytest.importorskip("mlflow")
pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")


_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "_mlflow_fixtures")
_LC_FIXTURE = os.path.join(_FIXTURES_DIR, "langchain_model.py")
_LG_FIXTURE = os.path.join(_FIXTURES_DIR, "langgraph_model.py")


def _payload_signature():
    """Object-typed signature shared by both fixtures → json input_mode."""
    from mlflow.models import ModelSignature  # type: ignore[import-not-found]
    from mlflow.types import DataType  # type: ignore[import-not-found]
    from mlflow.types.schema import (  # type: ignore[import-not-found]
        ColSpec,
        Object,
        Property,
        Schema,
    )

    return ModelSignature(
        inputs=Schema(
            [
                ColSpec(
                    name="payload",
                    type=Object(properties=[Property("text", DataType.string)]),
                )
            ]
        ),
        outputs=Schema([ColSpec(type=DataType.string)]),
    )


def _save_langchain_model(tmp: str) -> str:
    import mlflow  # type: ignore[import-not-found]

    model_dir = os.path.join(tmp, "lc_model")
    mlflow.langchain.save_model(  # type: ignore[attr-defined]
        lc_model=_LC_FIXTURE,
        path=model_dir,
        signature=_payload_signature(),
    )
    return model_dir


def _save_langgraph_model(tmp: str) -> str:
    import mlflow  # type: ignore[import-not-found]

    model_dir = os.path.join(tmp, "lg_model")
    mlflow.langchain.save_model(  # type: ignore[attr-defined]
        lc_model=_LG_FIXTURE,
        path=model_dir,
        signature=_payload_signature(),
    )
    return model_dir


def _assert_no_code_availability_warning(warn_mock) -> None:
    warned = [c.args[0] for c in warn_mock.call_args_list]
    for msg in warned:
        assert "serializes Python objects by reference" not in msg, (
            f"unexpected by-reference warning: {msg!r}"
        )
        assert "referenced from the Hugging Face Hub" not in msg, (
            f"unexpected Hub-reference warning: {msg!r}"
        )


def _assert_model_code_embedded(out_path: str, fixture_basename: str) -> None:
    with tarfile.open(out_path, "r") as tar:
        names = set(tar.getnames())
    expected = (
        f"variant_artifacts/extra_files/mlflow_model/{fixture_basename}"
    )
    assert expected in names, (
        f"expected {expected!r} in package, got {sorted(names)!r}"
    )


class TestLangChainModelsFromCode(unittest.TestCase):
    """Pure ``RunnableLambda`` round-trips through FNNX."""

    def test_round_trip(self):
        import mlflow  # type: ignore[import-not-found]

        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _save_langchain_model(tmp)
            out = os.path.join(tmp, "langchain.fnnx")

            with mock.patch("fnnx.extras.mlflow.console.warn") as warn_mock:
                package_mlflow_model(model_dir, out)

            _assert_model_code_embedded(out, "langchain_model.py")
            _assert_no_code_availability_warning(warn_mock)

            rt = Runtime(out)
            payload = {"text": "hello"}
            result = rt.compute({"data": {"payload": payload}}, {})

            # Compare to direct invocation through the embedded model.
            direct = mlflow.pyfunc.load_model(model_dir).predict(  # type: ignore[attr-defined]
                {"payload": payload}
            )

        self.assertIn("predictions", result)
        self.assertEqual(result["predictions"], direct)


class TestLangGraphModelsFromCode(unittest.TestCase):
    """Trivial ``StateGraph`` round-trips through FNNX."""

    def test_round_trip(self):
        import mlflow  # type: ignore[import-not-found]

        from fnnx.extras.mlflow import package_mlflow_model
        from fnnx.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _save_langgraph_model(tmp)
            out = os.path.join(tmp, "langgraph.fnnx")

            with mock.patch("fnnx.extras.mlflow.console.warn") as warn_mock:
                package_mlflow_model(model_dir, out)

            _assert_model_code_embedded(out, "langgraph_model.py")
            _assert_no_code_availability_warning(warn_mock)

            # Signature is an Object ColSpec → input_mode="json".
            import json

            with tarfile.open(out, "r") as tar:
                variant_config = json.loads(
                    tar.extractfile("variant_config.json").read().decode()  # type: ignore[union-attr]
                )
            self.assertEqual(
                variant_config["extra_values"]["fnnx_mlflow"]["input_mode"],
                "json",
            )

            rt = Runtime(out)
            payload = {"text": "hello"}
            result = rt.compute({"data": {"payload": payload}}, {})

            direct = mlflow.pyfunc.load_model(model_dir).predict(  # type: ignore[attr-defined]
                {"payload": payload}
            )

        self.assertIn("predictions", result)
        self.assertEqual(result["predictions"], direct)


if __name__ == "__main__":
    unittest.main()
