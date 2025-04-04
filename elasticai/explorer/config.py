from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any
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
    def __init__(self, config_path: Path):
        with open(config_path) as stream:
            try:
                self._original_yaml_dict: dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def dump_as_yaml(self, save_path: Path):
        """Creates a .yaml file of the current config.

        Args:
            save_path: The full or relative path to save config to.
        """

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w+") as ff:
            yaml.dump(
                yaml.safe_load(str(vars(self))), stream=ff, default_flow_style=False
            )

    def parse_optional(self):
        pass

    # TODO add parse_mandatory for single parameters with individual error message. 


class HWNASConfig(Config):
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self._original_yaml_dict = self._original_yaml_dict.get("HWNASConfig", {})
        self.parse_optional()

    def parse_optional(self):
        # set to default value, if yaml dict does not define a value
        self.host_processor: str = self._original_yaml_dict.get("host_processor", "cpu")
        self.max_search_trials: int = self._original_yaml_dict.get(
            "max_search_trials", 6
        )
        self.top_n_models: int = self._original_yaml_dict.get("top_n_models", 2)


class DeploymentConfig(Config):
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        # Get neccessary parameters for deployment with docker.
        self._original_yaml_dict: dict[Any, Any] = self._original_yaml_dict.get(
            "DeploymentConfig", {"Docker": {}}
        )

        # Get necessary parameters for connection with target.
        try:
            self.target_name: str = self._original_yaml_dict["target_name"]
            self.target_user: str = self._original_yaml_dict["target_user"]

        except KeyError:
            logger.info(
                "DeploymentConfig is not specified completely! Please specify or target connection is not possible."
            )
            exit(-1)

        self.parse_optional()

    def parse_optional(self):
        self.target_platform_name: str = self._original_yaml_dict.get(
            "target_platform_name", "rpi5"
        )

        self.docker = DockerParameter
        self._docker_yaml_dict = self._original_yaml_dict.get("Docker", {})
        self.docker.compiler_tag = self._docker_yaml_dict.get("compiler_tag", "cross")
        self.docker.path_to_dockerfile = Path(
            self._docker_yaml_dict.get(
                "path_to_dockerfile", ROOT_DIR / "docker" / "Dockerfile.picross"
            )
        )
        self.docker.build_context = Path(
            self._docker_yaml_dict.get("build_context", ROOT_DIR / "docker")
        )
        self.docker.compiled_library_path = Path(
            self._docker_yaml_dict.get("compiled_library_path", "./code/libtorch")
        )


class ModelConfig(Config):
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.parse_optional()

    def parse_optional(self):
        self._original_yaml_dict = self._original_yaml_dict.get("ModelConfig", {})
        self.model_type: str = self._original_yaml_dict.get("model_type", "MLP")
