import logging
import os
from pathlib import Path
import yaml

logger = logging.getLogger("explorer.config")


class Config:
    def __init__(self, config_path: Path):
        with open(config_path, encoding="utf-8") as stream:
            try:
                self.original_yaml_dict: dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def dump_as_yaml(self, save_path: Path):
        """Creates a .yaml file of the current config.

        Args:
            save_path: The full or relative path to save config to.
        """

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w+", encoding="utf-8") as ff:
            yaml.dump(
                yaml.safe_load(str(vars(self))), stream=ff, default_flow_style=False
            )


class HWNASConfig(Config):
    def __init__(self, config_path: Path):
        super().__init__(config_path)

        self.original_yaml_dict = self.original_yaml_dict.get("HWNASConfig", {})

        # set to default value, if yaml dict does not define a value
        self.host_processor: str = self.original_yaml_dict.get("host_processor", "cpu")
        self.max_search_trials: int = self.original_yaml_dict.get(
            "max_search_trials", 6
        )
        self.top_n_models: int = self.original_yaml_dict.get("top_n_models", 2)


class ConnectionConfig(Config):
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.original_yaml_dict = self.original_yaml_dict.get("ConnectionConfig", {})
        try:
            self.target_name: str = self.original_yaml_dict["target_name"]
            self.target_user: str = self.original_yaml_dict["target_user"]
        except KeyError:
            logger.info(
                "ConnectionConfig is not specified completely! Please specify or target connection is not possible."
            )
            exit(-1)


class ModelConfig(Config):
    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self.original_yaml_dict = self.original_yaml_dict.get("ModelConfig", {})
        self.model_type: str = self.original_yaml_dict["model_type"]
