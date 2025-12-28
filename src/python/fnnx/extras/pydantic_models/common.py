# ==============================================================
# This file was automatically copied from spec.
# DO NOT EDIT — changes here will be overwritten.
# ==============================================================

from pydantic import BaseModel, ConfigDict


class Empty(BaseModel):
    ...

    model_config = ConfigDict(extra="forbid")
