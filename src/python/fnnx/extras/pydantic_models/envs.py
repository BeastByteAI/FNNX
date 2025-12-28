# ==============================================================
# This file was automatically copied from spec.
# DO NOT EDIT — changes here will be overwritten.
# ==============================================================

from pydantic import BaseModel


class PipCondition(BaseModel):
    platform: list[str] | None = None
    os: list[str] | None = None
    accelerator: list[str] | None = None


class PipDependency(BaseModel):
    package: str
    extra_pip_args: str | None = None
    condition: PipCondition | None = None


class Python3_CondaPip(BaseModel):
    # `python3::conda_pip`
    python_version: str
    build_dependencies: list[str]
    dependencies: list[PipDependency]
    conda_channels: list[str] | None = None
