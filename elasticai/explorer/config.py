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
    compiled_library_path: Path
    path_to_dockerfile: Path
    build_context: Path


class Config:
    """ The Config Superclass for Elastic.AI.explorer.
    """
    def __init__(self, config_path: Path):
        with open(config_path) as stream:
            try:
                self._original_yaml_dict: dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def dump_as_yaml(self, save_path: Path):
        """ Creates a .yaml file of the current config.

        Args:
            save_path: The full or relative path to save config to.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w+") as ff:
            yaml.dump(
                yaml.safe_load(str(vars(self))), stream=ff, default_flow_style=False
            )

    def _parse_optional(self, parameter_name: str, default: Any) -> Any:
        """dict.get() wrapper"""
        return self._original_yaml_dict.get(parameter_name, default)

    def _parse_mandatory(self, parameter_name: str) -> Any:
        try:
            return self._original_yaml_dict[parameter_name]
        except KeyError:
            logger.info(
                f'The mandatory parameters of the {type(self).__name__} are not specified completely! Please specify parameter "{parameter_name}".'
            )
            exit(-1)


class HWNASConfig(Config):
    """ HWNASConfig that defines the HW-Nas Behavior and its excution on host.
    """
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self._original_yaml_dict = self._parse_optional("HWNASConfig", {})
        self.host_processor: str = self._parse_optional("host_processor", "auto")
        if self.host_processor == "auto":
            self.host_processor = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_search_trials: int = self._parse_optional("max_search_trials", 6)
        self.top_n_models: int = self._parse_optional("top_n_models", 2)


class DeploymentConfig(Config):
    """ The DeploymentConfig gives the necessary information to connect to the target-device and deploy model(s) on it.
    """
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        # Get neccessary parameters for deployment with docker.
        self._original_yaml_dict: dict[Any, Any] = self._parse_optional(
            "DeploymentConfig", {"Docker": {}}
        )

        # Get necessary parameters for connection with target.
        self.target_name: str = self._parse_mandatory("target_name")
        self.target_user: str = self._parse_mandatory("target_user")

        self.target_platform_name: str = self._parse_optional(
            "target_platform_name", "rpi5"
        )
        self.docker = DockerParameter
        self._docker_yaml_dict = self._parse_optional("Docker", {})
        self.docker.compiler_tag = self._parse_optional_docker("compiler_tag", "cross")
        self.docker.path_to_dockerfile = Path(
            self._parse_optional_docker(
                "path_to_dockerfile", ROOT_DIR / "docker" / "Dockerfile.picross"
            )
        )
        self.docker.build_context = Path(
            self._parse_optional_docker("build_context", ROOT_DIR / "docker")
        )
        self.docker.compiled_library_path = Path(
            self._parse_optional_docker("compiled_library_path", "./code/libtorch")
        )

    def _parse_optional_docker(self, parameter_name: str, default: Any) -> Any:
        """dict.get() wrapper"""
        return self._docker_yaml_dict.get(parameter_name, default)


class ModelConfig(Config):
    """ ModelConfig defines the type of deep neural network to search for.
    """
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self._original_yaml_dict = self._parse_optional("ModelConfig", {})
        self.model_type: str = self._parse_optional("model_type", "MLP")
