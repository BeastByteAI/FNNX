from fnnx.handlers._base import BaseHandler
from fnnx.handlers.local import LocalHandler, LocalHandlerConfig
from fnnx.device import DeviceMap
from typing import Any, Type


class Runtime:
    def __init__(
        self,
        bundle_path: str,
        handler: Type[BaseHandler] | None = None,
        handler_config: Any = None,
        device_map: str | DeviceMap | None = None,
        *args,
        **kwargs,
    ):
        if handler is None:
            handler = LocalHandler
            handler_config = (
                handler_config
                if isinstance(handler_config, LocalHandlerConfig)
                else LocalHandlerConfig()
            )

        if device_map is None:
            device_map = DeviceMap(accelerator="cpu", node_device_map={})
        elif isinstance(device_map, str):
            device_map = DeviceMap(accelerator=device_map, node_device_map={})
        self.handler: BaseHandler = handler(bundle_path, device_map, handler_config)

    def compute(self, inputs: dict, dynamic_attributes: dict):
        return self.handler.compute(inputs, dynamic_attributes)

    async def compute_async(self, inputs: dict, dynamic_attributes: dict):
        return await self.handler.compute_async(inputs, dynamic_attributes)
