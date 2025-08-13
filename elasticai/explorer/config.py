from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any
import torch
import yaml

from settings import ROOT_DIR

logger = logging.getLogger("explorer.config")


@dataclass
class DockerParameter:
    compiler_tag: str
    library_path: Path
    path_to_dockerfile: Path
    build_context: Path


class Config:
    def __init__(self, config_path: Path | None = None):
        if config_path:
            with open(config_path) as stream:
                try:
                    self._original_yaml_dict: dict = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            self._original_yaml_dict: dict = {}

    def dump_as_yaml(self, save_path: Path):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w+") as ff:
            yaml.dump(
                yaml.safe_load(str(vars(self))), stream=ff, default_flow_style=False
            )

    def _parse_optional(
        self, parameter_name: str, default: Any, category: str | None = None
    ) -> Any:
        if category:
            return self._original_yaml_dict.get(category, {}).get(
                parameter_name, default
            )
        else:
            return self._original_yaml_dict.get(parameter_name, default)

    def _parse_mandatory(self, parameter_name: str, category: str | None = None) -> Any:
        try:
            if category:
                return self._original_yaml_dict[category][parameter_name]
            else:
                return self._original_yaml_dict[parameter_name]
        except KeyError as err:
            logger.error(
                f'The mandatory parameters of the {type(self).__name__} are not specified completely! Please specify parameter "{parameter_name}" by defining a config.yaml.'
            )
            raise err


class HWNASConfig(Config):
    """HWNASConfig that defines the HW-Nas Behavior and its execution on host."""

    def __init__(self, config_path: Path | None = None):
        super().__init__(config_path)
        self._original_yaml_dict: dict = self._parse_optional("HWNASConfig", {})
        self.host_processor: str = self._parse_optional("host_processor", "auto")
        self.hw_constraints: dict = self._parse_optional("hw_constraints", {})
        self.search_algorithm: str = self._parse_optional("search_algorithm", "random")
        if self.host_processor == "auto":
            self.host_processor = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_search_trials: int = self._parse_optional("max_search_trials", 6)
        self.top_n_models: int = self._parse_optional("top_n_models", 2)
        self.n_estimation_epochs: int = self._parse_optional("n_estimation_epochs", 3)
        self.flops_weight: float = self._parse_optional("flops_weight", 2.0)


class DeploymentConfig(Config):
    """The DeploymentConfig gives the necessary information to connect to the target-device and deploy model(s) on it."""

    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self._original_yaml_dict: dict[Any, Any] = self._parse_mandatory(
            "DeploymentConfig"
        )

        self.target_name: str = self._parse_mandatory("target_name")
        self.target_user: str = self._parse_mandatory("target_user")
        self.target_platform_name: str = self._parse_optional(
            "target_platform_name", "rpi5"
        )
        self.serial_port: str = self._parse_optional(
            "serial_port", "/dev/ttyACM0"
        )

        self.docker = DockerParameter
        self.docker.compiler_tag = self._parse_optional(
            "compiler_tag", "cross", "Docker"
        )
        self.docker.path_to_dockerfile = Path(
            self._parse_optional(
                "path_to_dockerfile",
                ROOT_DIR / "docker" / "Dockerfile.picross",
                "Docker",
            )
        )
        self.docker.build_context = Path(
            self._parse_optional("build_context", ROOT_DIR / "docker", "Docker")
        )
        self.docker.library_path = Path(
            self._parse_optional("compiled_library_path", "./code/libtorch", "Docker")
        )
