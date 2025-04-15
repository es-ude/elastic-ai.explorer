import json
import jsonpickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.deployment.device_communication import Host
from elasticai.explorer.platforms.deployment.manager import HWManager
from elasticai.explorer.platforms.generator.generator import Generator


@dataclass
class HWPlatform:
    name: str
    info: str
    model_generator: Type[Generator]
    platform_manager: Type[HWManager]
    communication_protocol: Type[Host]
    compiler: Type[Compiler]


class KnowledgeRepository:
    def __init__(self):
        self.supported_hw_platforms = {}

    def register_hw_platform(self, platform: HWPlatform):
        self.supported_hw_platforms[platform.name] = platform

    def fetch_hw_info(self, name: str) -> HWPlatform:
        return self.supported_hw_platforms[name]

    def save_hw_platforms(self, path_to_repo: Path, add_new: bool = True):
        json_string = ""
        if add_new:
            self.load_hw_platforms(path_to_repo, load_all=True)
            json_string: str = str(jsonpickle.encode(self.supported_hw_platforms))

        else:
            json_string: str = str(jsonpickle.encode(self.supported_hw_platforms))
        with open(path_to_repo, "w+", encoding="utf-8") as f:
            f.write(json_string)

    def load_hw_platforms(
        self,
        path_to_repo: Path = Path(
            "elasticai/explorer/knowledge_repository/hw_platforms.json"
        ),
        load_all: bool = True,
        platform_name: None | str = None,
    ):
        with open(path_to_repo, "r+", encoding="utf-8") as f:
            json_string: str = str(json.dumps(json.load(f)))
        if load_all:
            hw_pfs: Any = jsonpickle.decode(json_string)

            for key, hw_pf in hw_pfs.items():
                self.register_hw_platform(hw_pf)
        elif platform_name:
            hw: HWPlatform = jsonpickle.decode(json_string)[platform_name]  # type: ignore

            self.register_hw_platform(hw)
