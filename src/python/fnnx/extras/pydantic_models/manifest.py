# ==============================================================
# This file was automatically copied from spec.
# DO NOT EDIT — changes here will be overwritten.
# ==============================================================

from pydantic import BaseModel, Field
from typing import Literal


class ModelIO(BaseModel):
    name: str
    content_type: str
    dtype: str
    tags: list[str] | None = None


class JSON(ModelIO):
    content_type: Literal["JSON"]


class NDJSON(ModelIO):
    content_type: Literal["NDJSON"]
    dtype: str = Field(
        ...,  # required field
        pattern=r"^(Array\[.+\]|NDContainer\[.+\])$",
        description="Must be in format 'Array[...]' or 'NDContainer[...]'",
    )
    shape: list[str | int]


class Var(BaseModel):
    name: str
    description: str
    tags: list[str] | None = None


class Manifest(BaseModel):
    variant: str

    name: str | None = None
    version: str | None = None
    description: str | None = None

    producer_name: str
    producer_version: str
    producer_tags: list[str]

    inputs: list[NDJSON | JSON]
    outputs: list[NDJSON | JSON]

    dynamic_attributes: list[Var]
    env_vars: list[Var]
