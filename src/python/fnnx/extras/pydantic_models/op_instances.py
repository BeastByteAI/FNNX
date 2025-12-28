# ==============================================================
# This file was automatically copied from spec.
# DO NOT EDIT — changes here will be overwritten.
# ==============================================================

from pydantic import BaseModel, StringConstraints
from typing import Union, Annotated


class OpIO(BaseModel):
    dtype: str
    shape: list[Union[int, str]]


class OpDynamicAttribute(BaseModel):
    name: str  # Source attribute name (external)
    default_value: str


class OpInstance(BaseModel):
    id: Annotated[str, StringConstraints(pattern="^[a-zA-Z0-9_]+$")]
    op: str
    inputs: list[OpIO]
    outputs: list[OpIO]
    attributes: dict
    dynamic_attributes: dict[
        str, OpDynamicAttribute
    ]  # {"<internal_name>": OpDynamicAttribute}


class OpInstances(BaseModel):
    ops: list[OpInstance]
