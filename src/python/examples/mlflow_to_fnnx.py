"""Hands-on demo: MLflow model -> FNNX package -> inspect -> run.

Builds a tiny MLflow model for one framework, converts it with
``fnnx.extras.mlflow.package_mlflow_model``, prints the package contents +
manifest/env/provenance, then runs it through ``fnnx.runtime.Runtime`` and
compares against the original model.

Usage (from src/python, with the deps for the chosen framework installed):

    python examples/mlflow_to_fnnx.py sklearn
    python examples/mlflow_to_fnnx.py torch
    python examples/mlflow_to_fnnx.py langchain
    python examples/mlflow_to_fnnx.py inspect _fnnx_demo/sklearn.fnnx   # inspect only

Each build writes <framework>.fnnx and the source MLflow dir under ./_fnnx_demo/
so you can poke at them afterwards (e.g. `tar -tf _fnnx_demo/sklearn.fnnx`).

The key thing the demo illustrates is how the runtime ``inputs`` dict is keyed,
which depends on the converter's chosen ``input_mode``:
  * columns mode  -> {column_name: 1-D array, ...}     (one key per signature column)
  * tensor mode   -> {"input": ndarray}                (single unnamed tensor)
                     or {name: ndarray, ...}            (named tensors)
  * json mode     -> {"data": <any JSON-able object>}  (passthrough / nested signatures)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile

from fnnx.extras.mlflow import package_mlflow_model
from fnnx.extras.reader import Reader
from fnnx.runtime import Runtime


OUT_DIR = os.path.join(os.getcwd(), "_fnnx_demo")


def _reset(model_dir: str) -> None:
    """mlflow.save_model refuses a non-empty target; clear it for re-runs."""
    shutil.rmtree(model_dir, ignore_errors=True)


def inspect(fnnx_path: str) -> None:
    """Print the package file tree + manifest/env/provenance via Reader."""
    print(f"\n=== inspect {fnnx_path} ===")
    with tarfile.open(fnnx_path, "r") as tar:
        print("-- files --")
        for name in sorted(tar.getnames()):
            print("  ", name)

    reader = Reader(fnnx_path)
    print("-- manifest --")
    print(reader.manifest.model_dump_json(indent=2))
    print("-- input_mode (variant_config) --")
    with tarfile.open(fnnx_path, "r") as tar:
        vc = json.loads(tar.extractfile("variant_config.json").read().decode())  # type: ignore[union-attr]
    print("  ", vc["extra_values"]["fnnx_mlflow"])
    if reader.pyenv is not None:
        print("-- pip dependencies --")
        for dep in reader.pyenv.dependencies:
            print("  ", dep.package)
    print("-- provenance (meta) --")
    for entry in reader.metadata:
        print("  id:", entry.id, "| tags:", entry.producer_tags)


def demo_sklearn() -> None:
    """RandomForest on a 2-column frame -> columns input_mode."""
    import mlflow
    import numpy as np
    import pandas as pd
    from mlflow.models.signature import infer_signature
    from sklearn.ensemble import RandomForestClassifier

    model_dir = os.path.join(OUT_DIR, "sklearn_model")
    out = os.path.join(OUT_DIR, "sklearn.fnnx")

    x = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0, 0.0]})
    y = [0, 1, 0, 1]
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
    sig = infer_signature(x, model.predict(x))
    _reset(model_dir)
    mlflow.sklearn.save_model(model, model_dir, signature=sig)  # type: ignore[attr-defined]

    package_mlflow_model(model_dir, out, name="sklearn-rf")
    inspect(out)

    rt = Runtime(out)
    inputs = {
        "a": x["a"].to_numpy(np.float64),
        "b": x["b"].to_numpy(np.float64),
    }
    res = rt.compute(inputs, {})
    print("\nfnnx predictions :", res["predictions"])
    print("sklearn predict  :", model.predict(x).tolist())


def demo_sklearn_array() -> None:
    """RandomForest trained on a bare numpy array -> tensor input_mode.

    Inferring the signature from a numpy ndarray (instead of a DataFrame)
    records a single unnamed TensorSpec, so the converter picks
    input_mode="tensor": the manifest has ONE input named "input" that takes
    the whole array. You call it with {"input": X} -- one key, the raw array --
    rather than one key per column. (The model still just sees model.predict(X).)
    """
    import mlflow
    import numpy as np
    from mlflow.models.signature import infer_signature
    from sklearn.ensemble import RandomForestClassifier

    model_dir = os.path.join(OUT_DIR, "sklearn_array_model")
    out = os.path.join(OUT_DIR, "sklearn_array.fnnx")

    x = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]], dtype=np.float64)
    y = [0, 1, 0, 1]
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
    sig = infer_signature(x, model.predict(x))  # ndarray in -> TensorSpec
    _reset(model_dir)
    mlflow.sklearn.save_model(model, model_dir, signature=sig)  # type: ignore[attr-defined]

    package_mlflow_model(model_dir, out, name="sklearn-rf-array")
    inspect(out)

    rt = Runtime(out)
    res = rt.compute({"input": x}, {})  # single "input" key holding the whole array
    print("\nfnnx predictions :", res["predictions"])
    print("sklearn predict  :", model.predict(x).tolist())


def demo_torch() -> None:
    """nn.Module in a separate module + code_paths -> tensor input_mode."""
    import mlflow
    import numpy as np
    import torch
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, TensorSpec

    # The module must live outside __main__ so torch's by-reference pickle can
    # resolve it; code_paths then embeds the source into the package so it
    # reloads on a clean machine too.
    mod_path = os.path.join(OUT_DIR, "tinynet_mod.py")
    with open(mod_path, "w") as f:
        f.write(
            "import torch\n"
            "from torch import nn\n"
            "N_FEATURES = 4\n"
            "class TinyNet(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.linear = nn.Linear(N_FEATURES, 2)\n"
            "    def forward(self, x):\n"
            "        return self.linear(x)\n"
        )
    if OUT_DIR not in sys.path:
        sys.path.insert(0, OUT_DIR)
    from tinynet_mod import N_FEATURES, TinyNet  # type: ignore[import-not-found]

    torch.manual_seed(0)
    model = TinyNet().eval()
    sample = np.random.default_rng(0).random((5, N_FEATURES)).astype(np.float32)

    model_dir = os.path.join(OUT_DIR, "torch_model")
    out = os.path.join(OUT_DIR, "torch.fnnx")
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float32"), [-1, N_FEATURES])]),
        outputs=Schema([TensorSpec(np.dtype("float32"), [-1, 2])]),
    )
    _reset(model_dir)
    # torch.save pickles the module by reference (its class must be importable);
    # code_paths embeds the source so the package reloads on a clean machine.
    mlflow.pytorch.save_model(  # type: ignore[attr-defined]
        model,
        model_dir,
        signature=signature,
        code_paths=[mod_path],
    )

    package_mlflow_model(model_dir, out, name="torch-tiny")
    inspect(out)

    rt = Runtime(out)
    res = rt.compute({"input": sample}, {})  # single unnamed tensor -> "input"
    with torch.no_grad():
        expected = model(torch.from_numpy(sample)).numpy()
    actual = np.asarray(res["predictions"], dtype=np.float32)
    print("\nmax abs diff vs torch forward:", float(np.abs(actual - expected).max()))


def demo_langchain() -> None:
    """models-from-code RunnableLambda with an Object signature -> json mode."""
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.types import DataType
    from mlflow.types.schema import ColSpec, Object, Property, Schema

    # models-from-code: a script that builds the runnable and registers it via
    # mlflow.models.set_model. The whole script is embedded in the package, so
    # it reloads without code_paths.
    script_path = os.path.join(OUT_DIR, "langchain_model.py")
    with open(script_path, "w") as f:
        f.write(
            "import mlflow\n"
            "from langchain_core.runnables import RunnableLambda\n"
            "def _shout(record):\n"
            "    payload = record.get('payload', record) if isinstance(record, dict) else record\n"
            "    text = payload.get('text', '') if isinstance(payload, dict) else str(payload)\n"
            "    return {'out': str(text).upper() + '!'}\n"
            "mlflow.models.set_model(RunnableLambda(_shout))\n"
        )

    signature = ModelSignature(
        inputs=Schema(
            [ColSpec(name="payload", type=Object(properties=[Property("text", DataType.string)]))]
        ),
        outputs=Schema([ColSpec(type=DataType.string)]),
    )

    model_dir = os.path.join(OUT_DIR, "langchain_model")
    out = os.path.join(OUT_DIR, "langchain.fnnx")
    _reset(model_dir)
    mlflow.langchain.save_model(  # type: ignore[attr-defined]
        lc_model=script_path, path=model_dir, signature=signature
    )

    package_mlflow_model(model_dir, out, name="langchain-shout")
    inspect(out)

    rt = Runtime(out)
    res = rt.compute({"data": {"payload": {"text": "hello"}}}, {})  # json mode -> "data"
    direct = mlflow.pyfunc.load_model(model_dir).predict({"payload": {"text": "hello"}})
    print("\nfnnx predictions :", res["predictions"])
    print("direct predict   :", direct)


DEMOS = {
    "sklearn": demo_sklearn,
    "sklearn-array": demo_sklearn_array,
    "torch": demo_torch,
    "langchain": demo_langchain,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in (*DEMOS, "inspect"):
        print(f"usage: {sys.argv[0]} {{{'|'.join(DEMOS)}|inspect <path>}}")
        raise SystemExit(2)

    os.makedirs(OUT_DIR, exist_ok=True)
    if sys.argv[1] == "inspect":
        inspect(sys.argv[2])
        return
    DEMOS[sys.argv[1]]()


if __name__ == "__main__":
    main()
