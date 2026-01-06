from abc import ABC, abstractmethod


class BaseEnvManager(ABC):
    def __init__(self, env_spec: dict, accelerator: str | None = None):
        pass

    @abstractmethod
    def ensure(self) -> None:
        pass

    @abstractmethod
    def python_cmd(self, argv: list[str]) -> list[str]:
        pass
