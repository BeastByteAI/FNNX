# ==============================================================
# This file was automatically copied from spec.
# DO NOT EDIT — changes here will be overwritten.
# ==============================================================

from pydantic import BaseModel


class MetaEntry(BaseModel):
    id: str
    producer: str
    producer_version: str
    producer_tags: list[str]
    payload: dict
