"""MLflow → FNNX packager.

This module converts an MLflow model directory (or any MLflow URI) into an
FNNX pyfunc-variant package. The local-directory path never imports
``mlflow``: it parses ``MLmodel`` (YAML) and the JSON-encoded signature
strings directly, mirroring what MLflow itself writes. ``mlflow`` is imported
lazily only where it is unavoidable — remote URI download and optional
``verify=True``.

Public API (added across Task 2/3): ``package_mlflow_model``,
``MLflowConverter``. Task 2 lands the readers and mapping helpers consumed by
the orchestrator (Task 3).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from fnnx.console import console
from fnnx.extras.pydantic_models.manifest import NDJSON, JSON, Var
from fnnx.extras.pydantic_models.meta import MetaEntry


_PERMISSIVE_INPUT_DTYPE = "ext::mlflow::input"
_PERMISSIVE_OUTPUT_DTYPE = "ext::mlflow::output"

_SINGLE_TENSOR_SENTINEL = "__single__"

# MLflow DataType → numpy dtype token used in FNNX Array[...].
_SCALAR_DTYPE_MAP: dict[str, str] = {
    "boolean": "bool",
    "integer": "int32",
    "long": "int64",
    "float": "float32",
    "double": "float64",
    "string": "string",
}


@dataclass
class MLModelInfo:
    """Normalized view of an MLflow ``MLmodel`` document.

    Both the mlflow-backed and YAML-fallback readers return this same shape so
    every downstream helper is reader-agnostic.
    """

    flavors: dict[str, dict]
    signature_inputs: list[dict] | None
    signature_outputs: list[dict] | None
    signature_params: list[dict] | None
    metadata: dict | None
    mlflow_version: str | None
    saved_input_example_info: dict | None
    raw_mlmodel: dict

    @property
    def loader_module(self) -> str | None:
        """Pyfunc loader module name (e.g. ``mlflow.sklearn``), if present."""
        pf = self.flavors.get("python_function") or {}
        loader = pf.get("loader_module")
        return loader if isinstance(loader, str) else None

    @property
    def primary_flavor(self) -> str:
        """Non-python_function flavor name, or ``python_function`` as fallback."""
        for name in self.flavors:
            if name != "python_function":
                return name
        return "python_function"


def _load_model_info(local_dir: str) -> MLModelInfo:
    """Read an MLflow model directory into the normalized ``MLModelInfo``.

    Prefers the mlflow library when importable (its own deserializers absorb
    format-version quirks); falls back to a direct YAML+JSON read when not.
    """
    try:
        import mlflow  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return _load_model_info_yaml(local_dir)
    return _load_model_info_mlflow(local_dir)


def _load_model_info_mlflow(local_dir: str) -> MLModelInfo:
    """Read MLmodel via the mlflow library (preferred)."""
    import mlflow.models  # type: ignore[import-not-found]

    m = mlflow.models.Model.load(local_dir)

    sig = m.signature
    signature_inputs: list[dict] | None = None
    signature_outputs: list[dict] | None = None
    signature_params: list[dict] | None = None
    if sig is not None:
        if sig.inputs is not None:
            signature_inputs = sig.inputs.to_dict()
        if sig.outputs is not None:
            signature_outputs = sig.outputs.to_dict()
        if sig.params is not None:
            signature_params = sig.params.to_dict()

    return MLModelInfo(
        flavors=dict(m.flavors or {}),
        signature_inputs=signature_inputs,
        signature_outputs=signature_outputs,
        signature_params=signature_params,
        metadata=m.metadata,
        mlflow_version=m.mlflow_version,
        saved_input_example_info=m.saved_input_example_info,
        raw_mlmodel=m.to_dict(),
    )


def _load_model_info_yaml(local_dir: str) -> MLModelInfo:
    """Read MLmodel by parsing the YAML directly (no mlflow import)."""
    import yaml  # pyyaml — declared in fnnx[extras]

    mlmodel_path = os.path.join(local_dir, "MLmodel")
    if not os.path.isfile(mlmodel_path):
        raise FileNotFoundError(
            f"MLflow MLmodel file not found at {mlmodel_path!r}"
        )
    with open(mlmodel_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"MLmodel at {mlmodel_path!r} did not parse to a mapping"
        )

    sig = raw.get("signature") or {}
    signature_inputs = _parse_signature_field(sig.get("inputs"))
    signature_outputs = _parse_signature_field(sig.get("outputs"))
    signature_params = _parse_signature_field(sig.get("params"))

    return MLModelInfo(
        flavors=dict(raw.get("flavors") or {}),
        signature_inputs=signature_inputs,
        signature_outputs=signature_outputs,
        signature_params=signature_params,
        metadata=raw.get("metadata"),
        mlflow_version=raw.get("mlflow_version"),
        saved_input_example_info=raw.get("saved_input_example_info"),
        raw_mlmodel=raw,
    )


def _parse_signature_field(field: Any) -> list[dict] | None:
    """Parse a signature field (`inputs`/`outputs`/`params`).

    MLflow writes these as JSON-encoded strings inside the MLmodel YAML, but
    we tolerate already-decoded lists too.
    """
    if field is None:
        return None
    if isinstance(field, str):
        decoded = json.loads(field)
    else:
        decoded = field
    if decoded is None:
        return None
    if not isinstance(decoded, list):
        raise ValueError(
            f"Signature field expected a JSON list, got {type(decoded).__name__}"
        )
    return decoded


def _map_input_schema(
    signature_inputs: list[dict] | None,
) -> tuple[list[NDJSON | JSON], str, dict, list[str]]:
    """Translate normalized MLflow signature inputs into FNNX manifest inputs.

    Returns ``(specs, input_mode, cfg_fragment, warnings)`` where
    ``cfg_fragment`` is merged into ``extra_values["fnnx_mlflow"]`` and
    ``warnings`` are user-facing strings to surface via ``console.warn``.
    """
    if signature_inputs is None:
        spec = JSON(
            name="data",
            content_type="JSON",
            dtype=_PERMISSIVE_INPUT_DTYPE,
        )
        cfg = {"input_mode": "passthrough"}
        warnings = [
            "MLflow model has no signature; using passthrough mode. "
            "Pass `input_specs=` to declare typed inputs."
        ]
        return [spec], "passthrough", cfg, warnings

    if not signature_inputs:
        raise ValueError("Signature inputs list is empty")

    is_all_tensor = all(_is_tensor_entry(e) for e in signature_inputs)
    if is_all_tensor:
        return _map_tensor_inputs(signature_inputs)

    is_all_simple_scalar = all(_is_simple_scalar_entry(e) for e in signature_inputs)
    if is_all_simple_scalar:
        return _map_columns_inputs(signature_inputs)

    # Nested ColSpec (object/array/map/any) or scalar binary/datetime → json.
    return _map_json_inputs()


def _is_tensor_entry(entry: dict) -> bool:
    return entry.get("type") == "tensor"


def _is_simple_scalar_entry(entry: dict) -> bool:
    """Scalar ColSpec with a DataType that maps to an Array[...] token."""
    t = entry.get("type")
    return isinstance(t, str) and t in _SCALAR_DTYPE_MAP


def _map_tensor_inputs(
    entries: list[dict],
) -> tuple[list[NDJSON | JSON], str, dict, list[str]]:
    specs: list[NDJSON | JSON] = []
    tensor_names: list[str] = []

    if len(entries) == 1 and not entries[0].get("name"):
        entry = entries[0]
        ts = entry.get("tensor-spec") or {}
        dtype_token = str(ts.get("dtype", "float32"))
        shape = _normalize_shape(ts.get("shape"))
        specs.append(
            NDJSON(
                name="input",
                content_type="NDJSON",
                dtype=f"Array[{dtype_token}]",
                shape=shape,
            )
        )
        tensor_names.append(_SINGLE_TENSOR_SENTINEL)
    else:
        for idx, entry in enumerate(entries):
            name = entry.get("name") or f"input_{idx}"
            ts = entry.get("tensor-spec") or {}
            dtype_token = str(ts.get("dtype", "float32"))
            shape = _normalize_shape(ts.get("shape"))
            specs.append(
                NDJSON(
                    name=name,
                    content_type="NDJSON",
                    dtype=f"Array[{dtype_token}]",
                    shape=shape,
                )
            )
            tensor_names.append(name)

    cfg = {"input_mode": "tensor", "tensor_names": tensor_names}
    return specs, "tensor", cfg, []


def _normalize_shape(shape: Any) -> list[str | int]:
    """Convert an MLflow shape list to FNNX shape (None/-1 stay as -1)."""
    if shape is None:
        return [-1]
    out: list[str | int] = []
    for dim in shape:
        if dim is None:
            out.append(-1)
        else:
            out.append(int(dim))
    return out


def _map_columns_inputs(
    entries: list[dict],
) -> tuple[list[NDJSON | JSON], str, dict, list[str]]:
    specs: list[NDJSON | JSON] = []
    column_order: list[str] = []
    for idx, entry in enumerate(entries):
        name = entry.get("name") or f"col_{idx}"
        token = _SCALAR_DTYPE_MAP[entry["type"]]
        specs.append(
            NDJSON(
                name=name,
                content_type="NDJSON",
                dtype=f"Array[{token}]",
                shape=[-1],
            )
        )
        column_order.append(name)
    cfg = {"input_mode": "columns", "column_order": column_order}
    return specs, "columns", cfg, []


def _map_json_inputs() -> tuple[list[NDJSON | JSON], str, dict, list[str]]:
    spec = JSON(
        name="data",
        content_type="JSON",
        dtype=_PERMISSIVE_INPUT_DTYPE,
    )
    cfg = {"input_mode": "json"}
    return [spec], "json", cfg, []


def _map_output_schema() -> tuple[list[NDJSON | JSON], list[str]]:
    """Outputs are always a single permissive JSON channel in v1."""
    spec = JSON(
        name="predictions",
        content_type="JSON",
        dtype=_PERMISSIVE_OUTPUT_DTYPE,
    )
    return [spec], []


def _map_params(
    signature_params: list[dict] | None,
) -> tuple[list[Var], list[str]]:
    """Translate ParamSpec entries into FNNX dynamic_attributes + their names."""
    if not signature_params:
        return [], []
    vars_: list[Var] = []
    names: list[str] = []
    for entry in signature_params:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"ParamSpec entry missing 'name': {entry!r}")
        ptype = entry.get("type", "?")
        default = entry.get("default", None)
        vars_.append(
            Var(
                name=name,
                description=(
                    f"MLflow predict() param '{name}' "
                    f"(type={ptype}, default={default!r})"
                ),
            )
        )
        names.append(name)
    return vars_, names


def _map_env(
    local_dir: str,
    flavors: dict[str, dict] | None = None,
) -> tuple[str | None, list[str], list[str]]:
    """Derive python version, build deps, and runtime deps for the FNNX env.

    Reads ``python_env.yaml`` and ``requirements.txt`` from the MLflow model
    dir. Falls back to the pyfunc flavor's ``python_version`` (then the
    current interpreter, signaled by ``None`` to the builder) when
    ``python_env.yaml`` is absent.
    """
    python_env_path = os.path.join(local_dir, "python_env.yaml")
    requirements_path = os.path.join(local_dir, "requirements.txt")

    python_version: str | None = None
    build_deps: list[str] = []

    if os.path.isfile(python_env_path):
        import yaml

        with open(python_env_path, "r") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            pv = data.get("python")
            if isinstance(pv, str) and pv:
                python_version = pv
            bd = data.get("build_dependencies")
            if isinstance(bd, list):
                build_deps = [str(x) for x in bd if isinstance(x, str)]

    if python_version is None and flavors is not None:
        pf = flavors.get("python_function") or {}
        pv = pf.get("python_version")
        if isinstance(pv, str) and pv:
            python_version = pv

    runtime_deps: list[str] = []
    if os.path.isfile(requirements_path):
        runtime_deps = _parse_requirements_txt(requirements_path)

    return python_version, build_deps, runtime_deps


def _parse_requirements_txt(path: str) -> list[str]:
    """Return one line per pip dependency.

    Skips blank lines, comments, and pip option / include lines (``-r``,
    ``-c``, anything starting with ``-``).
    """
    deps: list[str] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-"):
                continue
            deps.append(line)
    return deps


def _build_meta_payload(info: MLModelInfo, input_mode: str) -> dict:
    """Provenance payload for the ``mlflow-source`` meta entry."""
    return {
        "mlmodel": info.raw_mlmodel,
        "flavors": list(info.flavors.keys()),
        "loader_module": info.loader_module,
        "signature": {
            "inputs": info.signature_inputs,
            "outputs": info.signature_outputs,
            "params": info.signature_params,
        },
        "mlflow_version": info.mlflow_version,
        "input_mode": input_mode,
    }


def create_meta_callback(
    info: MLModelInfo,
    input_mode: str,
    producer_version: str,
) -> Callable:
    """Return a callback for ``PyfuncBuilder.create_meta_callback``.

    The callback writes a single ``mlflow-source`` MetaEntry capturing the
    original MLmodel content for lossless provenance.
    """
    primary_flavor = info.primary_flavor
    producer_tags = ["mlflow", primary_flavor]
    if info.mlflow_version:
        producer_tags.append(f"mlflow=={info.mlflow_version}")

    entry = MetaEntry(
        id="mlflow-source",
        producer="fnnx.extras.mlflow",
        producer_version=producer_version,
        producer_tags=producer_tags,
        payload=_build_meta_payload(info, input_mode),
    )

    def _callback(file_obj) -> None:
        file_obj.create_file(
            "meta.json",
            json.dumps([entry.model_dump()], indent=4, default=str),
        )

    return _callback


_PICKLE_LOADERS: set[str] = {
    "mlflow.pytorch",
    "mlflow.sklearn",
    "mlflow.pyfunc.model",
}

_NATIVE_LOADERS: set[str] = {
    "mlflow.onnx",
    "mlflow.xgboost",
    "mlflow.lightgbm",
}


def _self_containment_warnings(
    info: MLModelInfo, local_dir: str
) -> list[str]:
    """Emit warnings when the MLflow model dir may not be self-contained.

    The FNNX package inherits the underlying model's self-containment exactly
    (never worse, never better). These heuristics flag known risky shapes
    without ever blocking the build.
    """
    warnings: list[str] = []

    code_dir = os.path.join(local_dir, "code")
    has_code_dir = os.path.isdir(code_dir)

    pf = info.flavors.get("python_function") or {}
    model_code_path = pf.get("model_code_path")

    is_models_from_code = bool(model_code_path)
    is_self_contained_ish = has_code_dir or is_models_from_code

    loader = info.loader_module

    if loader == "mlflow.transformers":
        tf = info.flavors.get("transformers") or {}
        save_pretrained = tf.get("save_pretrained")
        if save_pretrained is False:
            warnings.append(
                "Transformers flavor was saved with `save_pretrained=False`: "
                "model weights are referenced from the Hugging Face Hub, not "
                "embedded in the model directory. The FNNX package will need "
                "Hub access at load time."
            )
        return warnings

    if loader in _NATIVE_LOADERS:
        return warnings

    if is_self_contained_ish:
        return warnings

    if loader in _PICKLE_LOADERS:
        warnings.append(
            f"Loader '{loader}' serializes Python objects by reference; the "
            "model/estimator class must be importable from the pinned "
            "requirements at load time. If it was defined in a local module, "
            "re-log with `code_paths=[…]`."
        )

    return warnings


def emit_warnings(messages: list[str]) -> None:
    """Surface a list of warning strings through fnnx.console."""
    for msg in messages:
        console.warn(msg)
