from pydantic import BaseModel


class PipCondition(BaseModel):
    platform: list[str] | None = None
    os: list[str] | None = None
    acclerator: list[str] | None = None


class PipDependency(BaseModel):
    package: str
    extra_pip_args: str | None = None
    condition: PipCondition | None = None


class Python3_CondaPip(BaseModel):
    python_version: str
    build_dependencies: list[str]
    dependencies: dict[str, str]
