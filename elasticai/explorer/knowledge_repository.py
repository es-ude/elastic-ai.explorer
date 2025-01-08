from dataclasses import dataclass
from typing import Type

from elasticai.explorer.platforms.deployment.manager import HWManager, PIHWManager
from elasticai.explorer.platforms.generator.generator import Generator, PIGenerator


@dataclass
class HWPlatform:
    name: str
    info: str
    model_generator: Type[Generator]
    platform_manager: Type[HWManager]


class KnowledgeRepository:
    def __init__(self):
        self.supported_hw_platforms = {}
        self.register_hw_platform(HWPlatform("rpi5", "Raspberry PI 5 with A76 processor and 8GB RAM", PIGenerator, PIHWManager))

    def register_hw_platform(self, platform: HWPlatform):
        self.supported_hw_platforms[platform.name] = platform

    def fetch_hw_info(self, name: str) -> HWPlatform:
        return self.supported_hw_platforms[name]
