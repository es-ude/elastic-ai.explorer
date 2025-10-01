from dataclasses import dataclass
from typing import Type

from elasticai.explorer.generator.deployment.compiler import Compiler
from elasticai.explorer.generator.deployment.device_communication import Host
from elasticai.explorer.generator.deployment.hw_manager import HWManager
from elasticai.explorer.generator.model_generator.model_generator import ModelGenerator


@dataclass
class Generator:
    name: str
    info: str
    model_generator: Type[ModelGenerator]
    platform_manager: Type[HWManager]
    communication_protocol: Type[Host]
    compiler: Type[Compiler]


class KnowledgeRepository:
    def __init__(self):
        self.supported_hw_platforms = {}

    def register_hw_platform(self, platform: Generator):
        self.supported_hw_platforms[platform.name] = platform

    def fetch_hw_info(self, name: str) -> Generator:
        return self.supported_hw_platforms[name]
