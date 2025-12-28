# ==============================================================
# This file was automatically copied from spec.
# DO NOT EDIT — changes here will be overwritten.
# ==============================================================

from pydantic import BaseModel
from typing import Literal
from pydantic_models.op_instances import OpInstance


class Opset(BaseModel):
    domain: str
    version: int


class ONNXAttributes(BaseModel):
    opsets: list[Opset]
    requires_ort_extensions: bool
    has_external_data: bool
    onnx_ir_version: int
    used_operators: dict[str, list[str]] | None = None


class ONNX_v1(OpInstance):
    op: Literal["ONNX_v1"]
    attributes: ONNXAttributes
