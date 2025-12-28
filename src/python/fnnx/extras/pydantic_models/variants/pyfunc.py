# ==============================================================
# This file was automatically copied from spec.
# DO NOT EDIT — changes here will be overwritten.
# ==============================================================

from pydantic import BaseModel
from typing import Any


class PyFuncVariant(BaseModel):

    pyfunc_classname: str
    extra_values: dict[str, Any] | None = None
