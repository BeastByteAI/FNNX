import json
import subprocess
from pydantic_models.op_instances import OpInstances
from pydantic_models.manifest import Manifest
from pydantic_models.variants.pipeline import PipelineVariant
from pydantic_models.variants.pyfunc import PyFuncVariant
from pydantic_models.ops.onnx import ONNX_v1
from pydantic_models.meta import MetaEntry
from pydantic_models.envs import Python3_CondaPip

SPEC_VERSION = "0.0.4"

manifest_json_schema = Manifest.model_json_schema()
ops_json_schema = OpInstances.model_json_schema()
meta_entry_json_schema = MetaEntry.model_json_schema()

combined_json_schema = {
    "version": SPEC_VERSION,
    "manifest": manifest_json_schema,
    "ops_entries": ops_json_schema,
    "meta_entry": meta_entry_json_schema,
    "envs": {},
    "ops": {},
    "variants": {},
}


with open("../manifest.json", "w+") as f:
    f.write(json.dumps(manifest_json_schema, indent=4))

with open("../ops.json", "w+") as f:
    f.write(json.dumps(ops_json_schema, indent=4))

with open("../meta_entry.json", "w+") as f:
    f.write(json.dumps(meta_entry_json_schema, indent=4))

with open("../env.json", "w+") as f:
    py_condapip_schema = Python3_CondaPip.model_json_schema()
    combined_json_schema["envs"]["python3::conda_pip"] = py_condapip_schema
    f.write(json.dumps(combined_json_schema["envs"]))


######### OPS

with open("../op_onnx_v1.json", "w+") as f:
    onnxv1 = ONNX_v1.model_json_schema()
    f.write(json.dumps(onnxv1, indent=4))
    combined_json_schema["ops"]["ONNX_v1"] = onnxv1

#############


############ VARIANTS

with open("../variant_pipeline.json", "w+") as f:
    pipeline = PipelineVariant.model_json_schema()
    f.write(json.dumps(pipeline, indent=4))
    combined_json_schema["variants"]["pipeline"] = pipeline

with open("../variant_pyfunc.json", "w+") as f:
    pyfunc = PyFuncVariant.model_json_schema()
    f.write(json.dumps(pyfunc, indent=4))
    combined_json_schema["variants"]["pyfunc"] = pyfunc

#############


with open("../combined.json", "w+") as f:
    f.write(json.dumps(combined_json_schema, indent=4))

spec_py_path = "../../../src/python/fnnx/spec.py"
with open(spec_py_path, "w") as f:
    f.write("# This file is auto generated and must not be modified manually!\n")
    f.write(f"schema = {combined_json_schema}")
subprocess.run(["black", spec_py_path])
