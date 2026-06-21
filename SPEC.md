# Proposals

## Problem

FNNX is a universal ML packaging format, and MLflow is the most widespread model
tracking/registry tool. Today, moving an MLflow model into FNNX is fully manual: the user
must write a `PyFunc` subclass, hand-declare manifest inputs/outputs, and re-declare the
runtime environment — even though the MLflow model already carries all of this information
(a typed signature, `python_env.yaml`/`requirements.txt`, and all artifacts).

We want a one-call conversion that packages **any** MLflow model — classic flavors
(sklearn, pytorch, onnx, …) and the newer GenAI "models-from-code" ones (ChatModel,
ChatAgent, ResponsesAgent, LangChain, …) — into an FNNX package, preserving the input
signature, inference params, and environment.

## Proposed solution

Add an MLflow packager to `fnnx.extras` that converts an MLflow model (given a local path
or a model URI) into an FNNX **pyfunc-variant** package:

1. **Universal interface via the `python_function` flavor.** Every loadable MLflow model
   exposes the `python_function` flavor and can be invoked through
   `mlflow.pyfunc.load_model(...).predict(...)`. This includes models-from-code/GenAI
   models. Building on pyfunc gives one conversion path for all flavors, present and
   future, instead of N per-flavor converters.
2. **Embed the MLflow model directory verbatim** under
   `variant_artifacts/extra_files/mlflow_model/`. This preserves the `MLmodel` file,
   weights/artifacts, the `code/` directory (when `code_paths` was used), and
   models-from-code Python files — everything MLflow needs to reload the model.
3. **Generate a static wrapper `__pyfunc__.py`** whose `warmup()` calls
   `mlflow.pyfunc.load_model()` on the embedded directory and whose `compute()` adapts
   FNNX inputs/outputs to the pyfunc `predict()` interface.
4. **Translate metadata mechanically:**
   - MLflow signature (ColSpec / TensorSpec schemas) → FNNX manifest `inputs`/`outputs`.
   - MLflow params schema → FNNX `dynamic_attributes`.
   - `python_env.yaml` + `requirements.txt` → FNNX `env.json` (`python3::conda_pip`).
   - Original `MLmodel` content → FNNX `meta.json` entry (provenance, lossless reference).

## Convert-time independence from MLflow (local paths)

Everything the converter needs to *package* a **local** MLflow model directory is already
on disk: the `MLmodel` file (YAML), the model signature (stored as JSON strings inside the
`MLmodel`), `python_env.yaml`, and `requirements.txt`. mlflow is only needed to *download* a
remote model URI and to *load/run* the model (verification, and the runtime wrapper itself).

Therefore, when `model_uri` points at a local directory, **packaging must succeed without
mlflow installed**. The converter reads `MLmodel` and the signature itself (a YAML read plus
`json.loads` of the signature strings) and the mapping helpers consume plain parsed JSON, so
no part of the local-path conversion imports mlflow. mlflow is still required — and lazily
imported with a clear error — only for remote-URI resolution and (optionally) verification.
The resulting package still depends on mlflow *at runtime*, exactly as before.

## Why this approach

- **Coverage over fidelity.** A per-flavor converter (e.g. sklearn → ONNX operator) would
  produce leaner, more portable packages but requires unbounded per-framework maintenance
  and can never cover arbitrary pyfunc/GenAI models. Wrapping via the pyfunc flavor covers
  the entire MLflow ecosystem with one code path. Per-flavor optimization can be layered
  on later without changing the user-facing API.
- **Reuses existing FNNX machinery.** The pyfunc variant, `PyfuncBuilder`, env spec, and
  meta entries already exist; the converter is a producer built on top of them, in line
  with the project's "builders live in `fnnx.extras`" pattern.
- **Cost:** the resulting package depends on `mlflow` at runtime (it is already pinned in
  the model's own requirements), and inherits MLflow's portability limits (next section).
  Convert-time mlflow is *not* required for local model directories — only for remote URIs
  and opt-in verification.

## Investigated gap: MLflow models are not always self-contained

The user's suspicion is **confirmed**. An MLflow model directory does not always contain
everything needed to reload the model:

- Pickle-based flavors (`mlflow.pytorch`, `mlflow.sklearn` with custom estimators,
  pickled `PythonModel` subclasses) serialize Python objects **by reference**: the class
  definition must be importable at load time. `torch.save(model)` stores the import path
  of the model class, not its source.
- cloudpickle captures classes defined in `__main__` **by value** (those models are
  self-contained), but classes defined in any other module **by reference**.
- MLflow's own mitigations are opt-in at save time: `code_paths=[...]` copies the listed
  source into the model's `code/` directory (added to `sys.path` at load), and
  `infer_code_paths=True` attempts to detect required local modules automatically. If the
  user saved the model without these and the class is not provided by a pip-installable
  requirement, the model cannot be loaded on a clean machine — by MLflow itself or by
  anything built on it.

**Consequence for FNNX conversion:** because the MLflow model directory is embedded
verbatim, the FNNX package inherits exactly the MLflow model's degree of
self-containment — never worse, never better. The converter therefore must:

1. Surface this honestly: emit a warning at convert time when heuristics suggest the
   package may not be self-contained (e.g. pickle-based flavor, no `code/` directory,
   loader module not among pinned requirements).
2. Offer an optional **verification step** that test-loads the converted package
   (optionally in a freshly created environment) to prove it is actually portable.

What the converter cannot do — and explicitly does not attempt — is recover source code
that was never saved. That failure mode is fixed at `mlflow.*.log_model()` time
(via `code_paths`), not at conversion time.

# Design

## Module and dependencies

New module **`fnnx.extras.mlflow`** (in `src/python/fnnx/extras/mlflow.py`), plus a static
runtime wrapper **`fnnx.extras._mlflow_wrapper`**
(`src/python/fnnx/extras/_mlflow_wrapper.py`).

Two dependency changes in `src/python/pyproject.toml`:

```toml
[project.optional-dependencies]
extras = ["pydantic>=2.0.0,<3.0.0", "pyyaml>=5.1"]   # pyyaml added: parse MLmodel/python_env.yaml
mlflow = ["mlflow>=3.4"]                             # 3.x only; 2.x is EOL for our purposes
test = [..., "pyyaml>=5.1", "mlflow>=3.4"]           # both added to the existing test group
```

- **`pyyaml`** is added to the **`extras`** group (which already carries `pydantic` for the
  builder). Packaging a **local** MLflow directory therefore needs only `fnnx[extras]` — no
  mlflow. `MLmodel` and `python_env.yaml` are YAML, read via `yaml.safe_load`.
- **`mlflow`** stays a separate **optional** group, required only for remote-URI resolution,
  verification, and at runtime. We target **mlflow 3.x only** (`>=3.4`); mlflow 2.x is
  considered EOL and is not supported. 3.x covers classic flavors, models-from-code, and
  ChatModel/ChatAgent/ResponsesAgent — the converter is flavor-agnostic anyway because it
  goes through the universal `python_function` flavor.
- **Framework dev/test deps** (test only): the e2e matrix adds `scikit-learn`, `pandas`,
  `xgboost`, `lightgbm`, `catboost`, `torch`, `langchain`, `langchain-core`, `langgraph`,
  `prophet`, and `tensorflow` to the `test` optional-dependency group (e.g.
  `uv add --optional test …`). None affect normal installs; each framework e2e test is gated
  with `pytest.importorskip(<framework>)`, so a missing heavy dep skips rather than fails.

The converter imports `mlflow` lazily (inside functions) **only on the paths that truly need
it** (remote URI download, verification). It raises a clear `ImportError` with install
instructions when one of those paths is taken without mlflow present. Local-directory
packaging never imports mlflow, and importing `fnnx.extras` never hard-requires it.

## Public API

```python
def package_mlflow_model(
    model_uri: str,
    output_path: str,
    *,
    name: str | None = None,
    version: str | None = None,
    description: str | None = None,
    input_specs: list[NDJSON | JSON] | None = None,
    output_specs: list[NDJSON | JSON] | None = None,
    extra_pip_dependencies: list[str] | None = None,
    producer_tags: list[str] | None = None,
    verify: bool = False,
) -> None
```

- `model_uri`: a local model directory **or** any MLflow URI (`runs:/…`, `models:/…`,
  `file:/…`, `s3://…`). Resolved to a local directory (see below).
- `output_path`: destination `.fnnx` file.
- `input_specs` / `output_specs`: explicit overrides. When provided they replace the
  auto-derived manifest I/O entirely (the user takes responsibility for matching the
  wrapper's expectations — only meaningful together with `input_mode` tags; primarily an
  escape hatch and the **required** path for no-signature models that want real typing).
- `extra_pip_dependencies`: appended to the env's runtime dependencies.
- `verify`: run an opt-in smoke test after building (see *Verification*).

The function is a thin wrapper over an internal `MLflowConverter` class (mirrors the
existing `PyfuncBuilder` style) that holds the parsed model and exposes `.save(path)`.

## Model resolution

```
if os.path.isdir(model_uri):
    local_dir = model_uri                      # no mlflow needed
else:
    # remote URI (runs:/…, models:/…, s3://…): requires mlflow
    import mlflow  # lazy; friendly ImportError if missing -> "pip install fnnx[mlflow]"
    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=<tempdir>)

info = _load_model_info(local_dir)             # reads MLmodel only; mlflow-optional (below)
```

The temp dir (when downloaded) is cleaned up after `.save()`. A non-directory `model_uri`
passed without mlflow installed raises a clear `ImportError` naming the remote-URI trigger.

## Reading the MLflow model (`_load_model_info`, mlflow-optional)

A single function normalizes the `MLmodel` into a plain dataclass that every downstream
helper consumes. It has two readers but **one normalized output**, so the mapping/provenance
layers never branch on whether mlflow is present:

```python
@dataclass
class MLModelInfo:
    flavors: dict[str, dict]               # MLmodel "flavors"
    signature_inputs: list[dict] | None    # parsed JSON list, or None when no signature
    signature_outputs: list[dict] | None
    signature_params: list[dict] | None
    metadata: dict | None                  # MLmodel "metadata"
    mlflow_version: str | None             # MLmodel "mlflow_version" (the saving version)
    saved_input_example_info: dict | None  # MLmodel "saved_input_example_info"
    raw_mlmodel: dict                      # full parsed MLmodel (for meta.json provenance)

def _load_model_info(local_dir: str) -> MLModelInfo:
    try:
        import mlflow  # noqa
        return _load_model_info_mlflow(local_dir)   # preferred when available
    except ImportError:
        return _load_model_info_yaml(local_dir)     # offline fallback
```

- **`_load_model_info_mlflow`** (preferred when mlflow is importable): `m =
  mlflow.models.Model.load(local_dir)`, then normalize via mlflow's own serializers so the
  output is byte-identical to the offline reader —
  `signature_inputs = m.signature.inputs.to_dict() if m.signature else None` (likewise
  `outputs`; `params = m.signature.params.to_dict() if … else None`), `flavors = m.flavors`,
  `metadata = m.metadata`, `mlflow_version = m.mlflow_version`,
  `saved_input_example_info = m.saved_input_example_info`, `raw_mlmodel = m.to_dict()`.
- **`_load_model_info_yaml`** (offline fallback, no mlflow): `raw = yaml.safe_load(open(<dir>/MLmodel))`.
  The signature lives under `raw["signature"]` as **JSON-encoded strings** —
  `signature_inputs = json.loads(sig["inputs"]) if "inputs" in sig else None` (likewise
  `outputs`, `params`); the rest come straight from the dict
  (`raw.get("flavors", {})`, `raw.get("metadata")`, `raw.get("mlflow_version")`,
  `raw.get("saved_input_example_info")`, `raw_mlmodel = raw`).

Because `Schema.to_json()`/`.to_dict()` is exactly what MLflow writes into the `MLmodel`
signature strings, both readers produce the same `signature_*` lists for a well-formed model
— enabling a direct equivalence test (Task 2). mlflow is *preferred* (it validates the file
and absorbs format/version quirks); the YAML reader is the offline path the local requirement
hinges on.

## Package shape produced

A standard **pyfunc-variant** FNNX package built with the existing `PyfuncBuilder`:

```
<output>.fnnx
├── manifest.json                 # derived inputs/outputs/dynamic_attributes + provenance
├── variant_config.json           # pyfunc_classname="MLflowModel", extra_values={...}
├── env.json                      # python3::conda_pip from python_env.yaml + requirements.txt
├── dtypes.json                   # ext:: dtypes for JSON I/O (permissive in v1)
├── meta.json                     # one entry: full MLmodel + signature provenance
├── variant_artifacts/
│   ├── __pyfunc__.py             # the static MLflowModel wrapper (shipped source)
│   └── extra_files/
│       └── mlflow_model/         # the ENTIRE MLflow model dir, verbatim
└── ops_artifacts/  meta_artifacts/  (empty .keep)
```

The MLflow directory is embedded verbatim via `builder.add_file(local_dir, "mlflow_model")`
— `File.copy` calls `tarfile.add(src, dst)`, which recurses directories, so the whole tree
(MLmodel, weights, `code/`, models-from-code `.py`, `requirements.txt`, …) lands under
`variant_artifacts/extra_files/mlflow_model/`. At runtime the wrapper resolves it with
`context.get_dirpath("mlflow_model")` (the Context scan indexes top-level dirs).

## The static wrapper (`__pyfunc__.py` / `MLflowModel`)

One fixed `PyFunc` subclass, identical for every converted model; all per-model behavior is
driven by `extra_values` (read via `context.get_value("fnnx_mlflow")`) plus the manifest.
It imports only `fnnx`, `mlflow`, and (lazily) `numpy`/`pandas` — never `fnnx.extras`.

`extra_values["fnnx_mlflow"]` schema:

```json
{
  "model_dir": "mlflow_model",
  "input_mode": "columns | tensor | json | passthrough",
  "column_order": ["sepal_len", "sepal_wid", ...],   // columns mode
  "tensor_names": ["input"] | ["a", "b"],            // tensor mode (["__single__"] = unnamed)
  "param_names": ["temperature", "max_tokens"]       // dynamic_attributes that are predict params
}
```

`warmup()`:
```python
model_dir = self.fnnx_context.get_dirpath(cfg["model_dir"])
self.model = mlflow.pyfunc.load_model(model_dir)
```

`compute(inputs, dynamic_attributes)`:
1. Build the predict payload from `inputs` per `input_mode`:
   - **columns** → `pandas.DataFrame({c: inputs[c] for c in column_order})`
     (each `inputs[c]` is a 1-D numpy array, already cast by the FNNX runtime).
   - **tensor** → single unnamed: `inputs[name]` (numpy ndarray);
     named: `{n: inputs[n] for n in tensor_names}`.
   - **json** / **passthrough** → `inputs["data"]` passed through unchanged.
2. `params = {n: dynamic_attributes[n] for n in param_names if n in dynamic_attributes}`.
3. `result = self.model.predict(payload, params=params) if params else self.model.predict(payload)`.
4. `return {"predictions": _to_jsonable(result)}`.

`compute_async` delegates to `compute`.

`_to_jsonable(x)` normalizes any predict() return into JSON-safe data (numpy
scalars→`.item()`, ndarrays→`.tolist()`, pandas DataFrame→`to_dict("records")`,
Series→`tolist()`, pydantic `BaseModel`→`model_dump()`, dataclass→`asdict`, dict/list
recurse, fallback `str(x)`). numpy/pandas are probed by module presence, not hard-imported.

## Signature → FNNX manifest (hybrid by schema kind)

Element-dtype mapping used below (FNNX `Array[...]` token must be a numpy dtype string the
runtime can `np.asarray(...).astype(token)`; `string`→`np.str_`):

| MLflow `DataType` | FNNX `Array[...]` token |
|---|---|
| boolean | `bool` |
| integer | `int32` |
| long | `int64` |
| float | `float32` |
| double | `float64` |
| string | `string` |
| binary, datetime | *(forces json mode — see below)* |

The mappers consume the normalized `signature_inputs: list[dict] | None` from
`MLModelInfo` — the same JSON entries MLflow serializes into the `MLmodel`. Entry shapes the
dispatch keys on (this is the on-disk format the YAML reader parses directly):

- **Scalar ColSpec:** `{"type": <DataType>, "name"?: str, "required"?: bool}` where
  `DataType ∈ {boolean, integer, long, float, double, string, binary, datetime}`.
- **TensorSpec:** `{"type": "tensor", "tensor-spec": {"dtype": <numpy dtype>, "shape": [...]},
  "name"?: str}`.
- **Nested ColSpec:** `{"type": "object", "properties": {...}}`, `{"type": "array", "items":
  {...}}`, `{"type": "map", "values": {...}}`, or `{"type": "any"}`.
- **ParamSpec:** `{"name": str, "type": <DataType>, "default": <value>, "shape": list | null}`.

Input dispatch on `signature_inputs`:

1. **`signature_inputs is None`** (no signature) → `input_mode="passthrough"`. One `JSON`
   input named `data`, dtype `ext::mlflow::input` registered as the permissive schema `{}`
   (accepts anything). Emit a warning that types could not be derived. `input_specs`
   override, if given, wins.
2. **Every entry has `type == "tensor"`** (TensorSpec schema) → `input_mode="tensor"`. One
   `NDJSON` input per tensor: `Array[<numpy dtype from "tensor-spec".dtype>]`, `shape` from
   `"tensor-spec".shape` with `-1` for variable dims. A single entry with no `name` → input
   name `input`, `tensor_names=["__single__"]`.
3. **All entries scalar ColSpec with a mappable `DataType`** → `input_mode="columns"`.
   One `NDJSON` input per column: name = entry `name` (or `col_<i>` if absent),
   `Array[<token>]`, `shape=[-1]`. `column_order` preserves entry order.
4. **Any entry nested (`object`/`array`/`map`/`any`) or scalar `binary`/`datetime`**
   → `input_mode="json"`. One `JSON` input `data`, dtype `ext::mlflow::input` permissive
   `{}`. Real enforcement is delegated to MLflow's own `predict()` signature checking; the
   exact MLflow schema is preserved losslessly in `meta.json`.

Output schema → **always a single `JSON` output `predictions`**, dtype
`ext::mlflow::output` (permissive `{}`). Rationale: predict() returns span ndarrays,
DataFrames, and arbitrary GenAI dicts; a single normalized JSON channel is universal and
lossless, and the FNNX local handler does not type-validate outputs anyway. Typed
per-column output mapping is a deliberate **non-goal for v1** (future enhancement).

Params (`signature_params: list[dict] | None`): each entry → a manifest `dynamic_attributes`
entry `Var(name=<param>, description="MLflow predict() param '<name>' (type=<type>, default=<default>)")`.
Param names are recorded in `param_names`. Defaults are left to MLflow (applied inside
predict when a param is omitted).

## Environment → `env.json`

Read from the embedded model dir:
- `python_env.yaml` → `python_version` (e.g. `3.9.8`), `build_dependencies`
  (`pip==`, `setuptools==`, `wheel==` lines, passed through to FNNX
  `build_dependencies`; the uv manager ignores them, conda installs them).
- `requirements.txt` → one `PipDependency(package=<line>)` per non-empty, non-comment line;
  `-r`/`-c`/`--` include/option lines are skipped (the referenced `requirements.txt` is the
  one we are already reading). `mlflow==<ver>` is normally present (MLflow auto-pins it),
  which is exactly what makes the wrapper importable at runtime.
- Append `fnnx==<fnnx_version>` (via `builder.add_fnnx_runtime_dependency()`) and any
  `extra_pip_dependencies`.
- If `python_env.yaml` is absent, fall back to the pyfunc flavor's `python_version`
  (`MLmodel.flavors.python_function.python_version`), else the current interpreter.

`PyfuncBuilder` is extended with a `python_version: str | None` constructor argument (used
in `_make_env` instead of the hard-coded module-level `PYTHON_VERSION` when set). This is
the only change to existing code.

## Provenance → `meta.json`

All fields below come from `MLModelInfo` (`raw_mlmodel`, `flavors`, `mlflow_version`,
normalized `signature_*`) — no mlflow import required. Via `create_meta_callback`, write one
`MetaEntry`:
```json
{
  "id": "mlflow-source",
  "producer": "fnnx.extras.mlflow",
  "producer_version": "<fnnx_version>",
  "producer_tags": ["mlflow", "<primary_flavor>", "mlflow==<mlflow_version>"],
  "payload": {
    "mlmodel": { ...full MLmodel dict... },
    "flavors": ["python_function", "sklearn", ...],
    "loader_module": "mlflow.sklearn",
    "signature": { "inputs": ..., "outputs": ..., "params": ... },
    "mlflow_version": "<x.y.z>",
    "input_mode": "columns"
  }
}
```
This keeps the original MLflow signature and metadata losslessly even when the FNNX-side
type is permissive.

## Self-containment heuristics (always-on warnings)

`_self_containment_warnings(info: MLModelInfo, local_dir) -> list[str]`, emitted via
`fnnx.console` (reads `info.flavors` / `loader_module`; no mlflow import):

- `code/` subdir present **or** pyfunc flavor `model_code_path` set (models-from-code) →
  considered self-contained-ish, **no** warning for code availability.
- Otherwise, if `loader_module` ∈ {`mlflow.pytorch`, `mlflow.sklearn`, `mlflow.pyfunc.model`}
  (pickle/cloudpickle object serialization) → warn: *"This flavor serializes Python objects
  by reference; the model/estimator class must be importable from the pinned requirements at
  load time. If it was defined in a local module, re-log with `code_paths=[…]`."*
- `loader_module` ∈ {`mlflow.onnx`, `mlflow.xgboost`, `mlflow.lightgbm`,
  `mlflow.transformers`} (native/self-describing artifacts) → no warning.
- `loader_module == mlflow.transformers` with a saved Hub reference only
  (`save_pretrained: false` in flavor config) → warn that weights are referenced from the
  Hub, not embedded.

Warnings never block the build; they document the inherited limitation described in
*Proposals → Investigated gap*.

## Verification (`verify=True`)

Verification inherently needs mlflow (it loads and runs the model). After `.save()`:

0. **Precheck:** if `mlflow` is not importable in the current process, emit a warning
   (*"verification skipped: mlflow is not installed in the current environment; install
   `fnnx[mlflow]` to verify"*) and **skip** the rest — the build still succeeds. This is the
   only place a missing mlflow does not error: an explicit `verify=True` degrades to a warning
   rather than failing the package that was already written.
1. `rt = fnnx.Runtime(output_path)` then access triggers `warmup()` →
   `mlflow.pyfunc.load_model(...)`. A failure here (e.g. missing class definition) is
   re-raised as `MLflowPackagingVerificationError` with the original cause and a hint that
   the model may not be self-contained.
2. If `info.saved_input_example_info` exists, load the example via mlflow
   (`example = mlflow.models.Model.load(local_dir).load_input_example(local_dir)`), convert
   it to FNNX inputs via the inverse of the wrapper's input assembly
   (`_example_to_fnnx_inputs(example, input_mode, cfg)`: DataFrame→columns dict / records
   list; ndarray/dict→tensor inputs; else→`{"data": example}`), and run one
   `rt.compute(inputs, dynamic_attributes={})`. Any exception → same error type.

Caveat documented in the error/text: a current-env smoke test proves portability only to
the extent the current env matches the target env. Fresh-env recreation (conda/uv) is **not**
part of v1.

## End-to-end test strategy (framework matrix)

The framework e2e tests follow one shared shape: train/build a tiny model with a fixed seed,
`mlflow.<flavor>.save_model(...)` (or models-from-code `log_model`) with an inferred
signature/input_example, `package_mlflow_model`, load via `fnnx.Runtime`, and assert the FNNX
output equals the original framework prediction for the same inputs. Constraints:

- **Determinism:** fixed seeds (`random_state` / `torch.manual_seed`), tiny data (~20 rows),
  few estimators, `eval()` for torch, `allow_writing_files=False` for catboost. No time- or
  order-dependent assertions.
- **Offline only:** LangChain/LangGraph models-from-code use a stub — a pure
  `RunnableLambda`, or `langchain_core.language_models.fake_chat_models.FakeListChatModel` —
  and a trivial `StateGraph`. No network or real LLM calls.
- **Graceful skips:** every test is gated by `pytest.importorskip("<framework>")` (and
  `pytest.importorskip("mlflow")`), so a missing heavy dep skips rather than fails.
- **Comparisons:** numeric outputs compared with tolerance (`np.allclose`); classifier
  labels compared exactly; the normalized `{"predictions": …}` channel is unwrapped first.

# Scenarios

## Scenario: Classic tabular model (sklearn, flat ColSpec signature)
**Given** an sklearn `RandomForestClassifier` saved with `mlflow.sklearn.save_model` and a
signature inferred from a pandas DataFrame with 4 double columns
**When** `package_mlflow_model(model_dir, "m.fnnx")` is called
**Then** the manifest has 4 `NDJSON` inputs `Array[float64] shape [-1]` named after the
columns, `input_mode="columns"`, a single `JSON` output `predictions`, the env lists
`scikit-learn==…` + `mlflow==…` + `fnnx==…`, and `meta.json` carries the full MLmodel.

## Scenario: End-to-end round-trip through the FNNX runtime
**Given** the package from the previous scenario
**When** it is loaded with `fnnx.Runtime("m.fnnx")` and `compute({col: [...], ...}, {})` is
called with one batch of rows
**Then** `warmup()` loads the embedded model via `mlflow.pyfunc.load_model`, the wrapper
assembles a DataFrame, calls `.predict`, and returns
`{"predictions": [...]}` whose values equal the original `model.predict` output for the same
rows.

## Scenario: Tensor signature (numpy TensorSpec)
**Given** a model whose signature is a single unnamed `TensorSpec(float32, [-1, 4])`
**When** it is packaged
**Then** `input_mode="tensor"`, there is one `NDJSON` input named `input` with
`Array[float32] shape [-1, 4]`, `tensor_names=["__single__"]`, and at runtime the wrapper
passes the numpy array straight to `.predict`.

## Scenario: GenAI / models-from-code (nested ColSpec, dict output)
**Given** a custom `mlflow.pyfunc.PythonModel` (or models-from-code script) whose input
signature contains an `Array`/`Object` column and whose `predict` returns a Python dict
**When** it is packaged
**Then** `input_mode="json"`, there is one `JSON` input `data` with permissive
`ext::mlflow::input`, the embedded dir contains the model-code `.py` at its root, and a
round-trip `compute({"data": <records>}, {})` returns `{"predictions": <dict>}` with the
dict JSON-normalized; the original nested signature is preserved in `meta.json`.

## Scenario: Gradient-boosting framework matrix (XGBoost, LightGBM, CatBoost)
**Given** a tiny classifier trained with each of `xgboost`, `lightgbm`, and `catboost` (fixed
seed, ~20 rows, a few estimators) and saved with the matching mlflow flavor
(`mlflow.xgboost`/`mlflow.lightgbm`/`mlflow.catboost`) with a signature inferred from a pandas
DataFrame
**When** each is packaged with `package_mlflow_model` and loaded via `fnnx.Runtime`
**Then** `input_mode="columns"`, a round-trip `compute` reproduces the original
`model.predict` labels exactly, the env pins the framework + `mlflow==…` + `fnnx==…`, and
**no** self-containment warning is emitted (native, self-describing artifacts).

## Scenario: PyTorch model with nn.Module defined outside `__main__`, saved with code_paths
**Given** a `torch.nn.Module` subclass defined in a **separate importable module**
(`tests/_mlflow_fixtures/torch_net.py`, not `__main__`), saved with
`mlflow.pytorch.save_model(..., code_paths=[<that file>])` and a `TensorSpec(float32, [-1, n])`
signature
**When** it is packaged and loaded via `fnnx.Runtime`
**Then** `input_mode="tensor"`, the embedded `mlflow_model/code/` directory contains the
module source, **no** code-availability warning is emitted, and a round-trip `compute`
reproduces the original `model(x)` outputs within `np.allclose` tolerance.

## Scenario: PyTorch nn.Module outside `__main__`, saved without code_paths → warning
**Given** the same `nn.Module` saved with `mlflow.pytorch.save_model` **without** `code_paths`
**When** it is packaged
**Then** the build succeeds, the embedded dir has **no** `code/` directory, and a
self-containment warning is emitted naming the by-reference pickle flavor and recommending
re-logging with `code_paths`.

## Scenario: LangChain models-from-code (offline) round-trips
**Given** a models-from-code script that builds a deterministic, offline LangChain runnable
(a pure `RunnableLambda`, or a prompt piped to `FakeListChatModel`) and registers it via
`mlflow.models.set_model`, logged with `mlflow.langchain.log_model(lc_model=<script>, …)`
**When** it is packaged and loaded via `fnnx.Runtime`
**Then** the embedded dir contains the model-code `.py`, no code-availability warning is
emitted (models-from-code sets `model_code_path`), and a round-trip `compute` returns
`{"predictions": …}` equal to invoking the runnable directly on the same input.

## Scenario: LangGraph models-from-code (offline) round-trips
**Given** a models-from-code script that builds a trivial deterministic `langgraph.StateGraph`
(a single node doing pure string/state manipulation, no LLM) registered via
`mlflow.models.set_model` and logged as a models-from-code pyfunc
**When** it is packaged and loaded via `fnnx.Runtime`
**Then** the model-code `.py` is embedded, the appropriate `input_mode` (`json`) is selected,
and a round-trip `compute` reproduces the graph's output for the same input.

## Scenario: Prophet time-series model (datetime column → json mode)
**Given** a `prophet` model fit on a tiny dataframe with a `ds` **datetime** column and a `y`
target, saved with `mlflow.prophet.save_model` and an inferred signature
**When** it is packaged and loaded via `fnnx.Runtime`
**Then** because the signature contains a `datetime` column, `input_mode="json"` is selected
(exercising the datetime branch with a real model), the original MLflow signature is preserved
in `meta.json`, and a round-trip `compute` returns forecasts equal to the model's own
`predict` for the same future dataframe.

## Scenario: Keras functional model with two named inputs (named tensor mode)
**Given** a small `tensorflow`/Keras functional model with two named inputs `a` and `b`
(fixed seed), saved with `mlflow.tensorflow.save_model` and a TensorSpec signature carrying
both names
**When** it is packaged and loaded via `fnnx.Runtime`
**Then** `input_mode="tensor"` with `tensor_names=["a", "b"]`, the manifest has one `NDJSON`
input per named tensor, and a round-trip `compute({"a": …, "b": …}, {})` reproduces the
model's outputs within `np.allclose` tolerance.

## Scenario: Transformers flavor with save_pretrained=false → Hub-reference warning
**Given** an `MLmodel` whose `transformers` flavor config sets `save_pretrained: false`
(weights referenced from the Hugging Face Hub, not embedded)
**When** the model directory is packaged
**Then** a self-containment warning is emitted stating the weights are referenced from the Hub
rather than bundled, and the build still succeeds. Verified offline with a synthetic `MLmodel`
(no `transformers` install or network required).

## Scenario: Inference params → dynamic attributes
**Given** a model with a params schema `[ParamSpec("temperature", double, 0.7),
ParamSpec("max_tokens", long, 256)]`
**When** it is packaged and run as
`compute({"data": ...}, {"temperature": 0.1})`
**Then** the manifest `dynamic_attributes` contains `temperature` and `max_tokens`, and the
wrapper calls `model.predict(payload, params={"temperature": 0.1})` (omitted params fall
back to MLflow defaults).

## Scenario: Model without a signature
**Given** an MLflow model saved without a signature
**When** it is packaged with no `input_specs` override
**Then** a warning is logged, `input_mode="passthrough"`, the manifest has a single
permissive `JSON` input `data` and `JSON` output `predictions`, and a round-trip passing the
raw payload under `"data"` still works.

## Scenario: Non-self-contained pickle model — convert-time warning
**Given** a `mlflow.pytorch` model saved **without** `code_paths`, whose `nn.Module` class
lives in a local module
**When** it is packaged
**Then** the build succeeds and a warning is emitted stating the class must be importable
from the pinned requirements and recommending re-logging with `code_paths`; the embedded dir
matches the source exactly (no `code/` directory).

## Scenario: Non-self-contained model — verification fails loudly
**Given** the same pytorch model and `verify=True`, run in an environment where the model
class is **not** importable
**When** `package_mlflow_model(..., verify=True)` is called
**Then** the `.fnnx` is written, then verification raises
`MLflowPackagingVerificationError` whose message names the underlying import error and notes
the model is likely not self-contained.

## Scenario: Self-contained model — verification passes with input example
**Given** an sklearn model saved **with** an `input_example` and `verify=True`
**When** it is packaged
**Then** verification loads the package via `fnnx.Runtime`, converts the saved input example
to FNNX inputs, runs one `compute`, and returns without error.

## Scenario: Model URI instead of a local path
**Given** `model_uri="models:/my_model/3"` (or `runs:/<id>/model`) and `mlflow` installed
**When** `package_mlflow_model(model_uri, "m.fnnx")` is called
**Then** the model is downloaded via `mlflow.artifacts.download_artifacts` to a temp dir,
packaged identically to the local-path case, and the temp dir is cleaned up afterward.

## Scenario: Local model directory packaged without mlflow installed
**Given** a local MLflow model directory (sklearn, with a ColSpec signature) and an
environment where `import mlflow` fails (simulated by `sys.modules["mlflow"] = None`)
**When** `package_mlflow_model(local_dir, "m.fnnx")` is called (no `verify`)
**Then** packaging succeeds without importing mlflow: `MLmodel` and the signature are read via
the YAML fallback, the manifest/env/meta are produced as in the sklearn scenario, the embedded
`mlflow_model/` dir matches the source, and the resulting `.fnnx` loads as a valid package.

## Scenario: YAML reader equals mlflow reader
**Given** the same local model directory
**When** `_load_model_info` runs once with mlflow importable and once with it blocked
**Then** both produce an `MLModelInfo` with equal `signature_inputs`, `signature_outputs`,
`signature_params`, and `flavors`, so the downstream manifest is identical either way.

## Scenario: Remote model URI without mlflow → ImportError
**Given** `model_uri="models:/my_model/3"` and no `mlflow` installed
**When** `package_mlflow_model(model_uri, "m.fnnx")` is called
**Then** a clear `ImportError` is raised naming the remote-URI trigger and instructing
`pip install fnnx[mlflow]`; importing `fnnx.extras` itself does not fail.

## Scenario: verify=True without mlflow → warn and skip
**Given** a local model directory, `verify=True`, and no `mlflow` importable in the current
process
**When** `package_mlflow_model(local_dir, "m.fnnx", verify=True)` is called
**Then** the `.fnnx` is written, a warning states verification was skipped because mlflow is
not installed, and no error is raised.

## Scenario: Binary/datetime column forces json mode
**Given** a ColSpec signature mixing `double` columns with a `datetime` column
**When** it is packaged
**Then** because `datetime` has no FNNX `Array` token, `input_mode="json"` is used (single
permissive `data` input) rather than per-column NDJSON.

# Tasks

- [x] **Task 1 — Static runtime wrapper + builder `python_version` override.**
  - [x] Add `src/python/fnnx/extras/_mlflow_wrapper.py` defining `MLflowModel(PyFunc)`:
        `warmup` (load via `mlflow.pyfunc.load_model(context.get_dirpath(cfg["model_dir"]))`),
        `compute` (input assembly per `input_mode`, param extraction, predict, output
        normalization), `compute_async` (delegates), and the module-level `_to_jsonable`
        helper. Import only `fnnx`, `mlflow` (lazy), `numpy`/`pandas` (probed).
  - [x] Extend `PyfuncBuilder.__init__` in `src/python/fnnx/extras/builder.py` with
        `python_version: str | None = None`, stored and used in `_make_env` when set.
  - [x] Tests in `src/python/tests/test_mlflow_wrapper.py`: import `MLflowModel`, inject a
        fake context + fake `self.model` (records the payload, returns canned values), and
        assert input assembly for columns/tensor/json/passthrough, param passing, and
        `_to_jsonable` for numpy/pandas/dict/pydantic/dataclass/fallback. Test builder
        `python_version` flows into `env.json`. Guard mlflow-dependent paths with
        `pytest.importorskip` only where a real mlflow object is needed (wrapper logic uses
        a fake model, so most tests need no mlflow).

- [x] **Task 2 — MLmodel reader (mlflow-optional) + mapping helpers in `fnnx/extras/mlflow.py`.**
  - [x] Define the `MLModelInfo` dataclass and `_load_model_info(local_dir)` with the two
        readers: `_load_model_info_mlflow` (preferred when `import mlflow` succeeds; normalize
        via `Model.load(...)` + `Schema.to_dict()`) and `_load_model_info_yaml` (offline
        fallback: `yaml.safe_load` the `MLmodel`, `json.loads` the signature strings). Both
        return the same normalized `signature_*` lists / `flavors` / `raw_mlmodel`.
  - [x] `_map_input_schema(signature_inputs: list[dict] | None) -> (specs, input_mode,
        cfg_fragment, warnings)` implementing the hybrid dispatch + dtype table on the
        normalized JSON entries (columns/tensor/json/passthrough, binary/datetime → json).
  - [x] `_map_params(signature_params: list[dict] | None) -> (list[Var], param_names)`.
  - [x] `_map_env(local_dir) -> (python_version, build_deps, runtime_deps)` parsing
        `python_env.yaml` (via `yaml.safe_load`) + `requirements.txt` (skip
        comments/`-r`/`-c`/option lines). No mlflow.
  - [x] `_build_meta_payload(info: MLModelInfo, input_mode) -> dict` and a
        `create_meta_callback` factory writing the `mlflow-source` `MetaEntry`.
  - [x] `_self_containment_warnings(info: MLModelInfo, local_dir) -> list[str]` per the
        heuristics table (reads `info.flavors`/`loader_module`).
  - [x] Register `ext::mlflow::input` / `ext::mlflow::output` permissive `{}` dtypes when
        json/passthrough modes are used.
  - [x] Tests in `src/python/tests/test_mlflow_mapping.py`: feed the mappers hand-built
        signature JSON lists directly (columns/tensor/json/passthrough/binary-datetime, params)
        — **no mlflow needed**. Add synthetic temp dirs/files for `_map_env` and the
        self-containment heuristics, including the `mlflow.transformers` flavor with
        `save_pretrained: false` (synthetic `flavors` dict) asserting the Hub-reference warning
        fires, and a native flavor asserting no warning. Add a `_load_model_info_yaml` test
        against a hand-authored `MLmodel` fixture, and an equivalence test
        (`pytest.importorskip("mlflow")`) asserting the mlflow and YAML readers produce equal
        `MLModelInfo` for a saved model.

- [x] **Task 3 — `package_mlflow_model` orchestrator + packaging + pyproject deps.**
  - [x] In `src/python/pyproject.toml`: add `pyyaml>=5.1` to the `extras` group, and add both
        `pyyaml>=5.1` and a `mlflow>=3.4` optional group + `mlflow>=3.4` to the `test`
        group.
  - [x] Implement `MLflowConverter` and `package_mlflow_model` in `fnnx/extras/mlflow.py`:
        resolve the model — **local dir → use directly, no mlflow import**; **non-local URI →
        lazy `import mlflow` (friendly `ImportError` if missing) + `download_artifacts`**.
        Call `_load_model_info` (Task 2) and the mappers, drive `PyfuncBuilder`
        (`PyFuncSpec(_mlflow_wrapper.__file__, "MLflowModel")`, `add_input`/`add_output`,
        `add_dynamic_attribute`, `define_dtype`, `set_extra_values({"fnnx_mlflow": cfg})`,
        `add_file(local_dir, "mlflow_model")`, env deps, `python_version`,
        `create_meta_callback`), then `save`. Emit self-containment warnings. Clean up temp
        dir (only created for URIs).
  - [x] Export `package_mlflow_model` from `fnnx.extras.mlflow`.
  - [x] Integration test in `src/python/tests/test_mlflow_e2e.py`: train a tiny sklearn
        model (fixed seed, ~20 rows) with an inferred signature, `mlflow.sklearn.save_model`
        to a temp dir, `package_mlflow_model`, load with `fnnx.Runtime`, run `compute`, and
        assert outputs equal the original `model.predict`. `pytest.importorskip("mlflow")`.
  - [x] No-mlflow test in the same file: save a model with mlflow, then set
        `sys.modules["mlflow"] = None` (monkeypatch) and assert `package_mlflow_model(local_dir,
        out)` (no `verify`) succeeds, produces a loadable package, and that a non-local
        `model_uri` under the same condition raises `ImportError`. Restore `sys.modules` after.

- [x] **Task 4 — Verification (`verify=True`) + input-example conversion.**
  - [x] Implement `MLflowPackagingVerificationError`, the `verify=True` smoke test with the
        mlflow precheck (if `mlflow` is not importable → warn and skip, build still succeeds),
        the Runtime load + optional input-example `compute`, and
        `_example_to_fnnx_inputs(example, input_mode, cfg)`. Load the input example via
        `mlflow.models.Model.load(local_dir).load_input_example(local_dir)`.
  - [x] Tests in `src/python/tests/test_mlflow_verify.py`: (a) sklearn model saved with an
        `input_example` → `verify=True` succeeds; (b) `_example_to_fnnx_inputs` unit tests
        for DataFrame/ndarray/dict/passthrough; (c) a model whose loader raises (simulate by
        an unimportable custom `PythonModel` / monkeypatched `load_model`) →
        `MLflowPackagingVerificationError`; (d) `verify=True` with `sys.modules["mlflow"] =
        None` → no error, a skip warning is emitted, package still written. `pytest.importorskip("mlflow")`
        for (a)–(c).

- [x] **Task 5 — Broader end-to-end coverage (GenAI/custom, tensor, no-signature).**
  - [x] Integration tests in `src/python/tests/test_mlflow_variants.py`:
        - [x] Custom `mlflow.pyfunc.PythonModel` returning a dict (json `input_mode`, dict
              output normalization) round-trips through `fnnx.Runtime`.
        - [x] A model with a `TensorSpec` signature (tensor `input_mode`) round-trips.
        - [x] A model saved without a signature (passthrough `input_mode`) round-trips and a
              warning is emitted.
        - [x] A params-schema model: assert `dynamic_attributes` mapping and that a passed
              dynamic attribute reaches `predict(..., params=...)`.
        `pytest.importorskip("mlflow")`.

- [x] **Task 6 — Gradient-boosting framework e2e (XGBoost, LightGBM, CatBoost).**
  - [x] Add `xgboost`, `lightgbm`, `catboost`, `pandas`, and `scikit-learn` to the `test`
        optional-dependency group in `src/python/pyproject.toml`.
  - [x] Integration tests in `src/python/tests/test_mlflow_frameworks.py` (one parametrized
        test or one per framework): train a tiny classifier (fixed seed, ~20 rows, few
        estimators; `allow_writing_files=False` for catboost) on a pandas DataFrame, infer a
        signature, `mlflow.<flavor>.save_model`, `package_mlflow_model`, load via
        `fnnx.Runtime`, assert `input_mode="columns"`, predictions match the original
        `model.predict` exactly, env pins the framework, and no self-containment warning is
        emitted. `pytest.importorskip("mlflow")` + `pytest.importorskip("<framework>")`.

- [x] **Task 7 — PyTorch e2e incl. nn.Module outside `__main__`.**
  - [x] Add `torch` to the `test` optional-dependency group in `src/python/pyproject.toml`.
  - [x] Add the fixture module `src/python/tests/_mlflow_fixtures/torch_net.py` defining a
        tiny `nn.Module` (e.g. one `nn.Linear`), and an `__init__.py` if needed for import.
  - [x] Integration tests in `src/python/tests/test_mlflow_torch.py`
        (`pytest.importorskip("torch")` + mlflow):
        - [x] Save the model with `mlflow.pytorch.save_model(..., code_paths=[torch_net.py])`
              and a `TensorSpec(float32, [-1, n])` signature → package → `fnnx.Runtime`
              round-trip equals `model(x)` (`np.allclose`); assert `mlflow_model/code/` is
              embedded and **no** code-availability warning.
        - [x] Save the **same** model **without** `code_paths` → assert a self-containment
              warning is emitted and the embedded dir has no `code/` directory.

- [ ] **Task 8 — LangChain / LangGraph models-from-code e2e (offline).**
  - [ ] Add `langchain`, `langchain-core`, and `langgraph` to the `test` optional-dependency
        group in `src/python/pyproject.toml`.
  - [ ] Add models-from-code fixture scripts under `src/python/tests/_mlflow_fixtures/`
        (`langchain_model.py`, `langgraph_model.py`), each building a deterministic **offline**
        runnable/graph (pure `RunnableLambda` or `FakeListChatModel`; trivial `StateGraph` with
        a pure node) and calling `mlflow.models.set_model(...)`.
  - [ ] Integration tests in `src/python/tests/test_mlflow_langchain.py`
        (`pytest.importorskip` for `mlflow`, `langchain_core`, `langgraph`): log each via
        models-from-code, `package_mlflow_model`, load via `fnnx.Runtime`, and assert the
        embedded model-code `.py` is present, no code-availability warning, and a round-trip
        `compute` reproduces direct invocation of the runnable/graph on the same input.

- [ ] **Task 9 — Prophet time-series e2e (datetime → json mode).**
  - [ ] Add `prophet` to the `test` optional-dependency group in `src/python/pyproject.toml`.
  - [ ] Integration test in `src/python/tests/test_mlflow_prophet.py`
        (`pytest.importorskip("prophet")` + mlflow): fit a tiny prophet model on a small
        fixed `ds`/`y` dataframe, `mlflow.prophet.save_model` with an inferred signature,
        `package_mlflow_model`, load via `fnnx.Runtime`, assert `input_mode="json"` (driven by
        the `ds` datetime column), and that a round-trip forecast equals the model's own
        `predict` on the same future dataframe.

- [ ] **Task 10 — TensorFlow/Keras named multi-input e2e (named tensor mode).**
  - [ ] Add `tensorflow` to the `test` optional-dependency group in
        `src/python/pyproject.toml`.
  - [ ] Integration test in `src/python/tests/test_mlflow_tensorflow.py`
        (`pytest.importorskip("tensorflow")` + mlflow): build a small Keras functional model
        with two named inputs `a`/`b` (`tf.random.set_seed`), `mlflow.tensorflow.save_model`
        with a TensorSpec signature carrying both names, `package_mlflow_model`, load via
        `fnnx.Runtime`, assert `input_mode="tensor"` with `tensor_names=["a", "b"]` and one
        `NDJSON` input per named tensor, and that `compute({"a": …, "b": …}, {})` matches the
        model output within `np.allclose`.
