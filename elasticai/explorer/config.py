import datetime
import logging
import os
from pathlib import Path

import yaml
from settings import MAIN_EXPERIMENT_DIR

logger = logging.getLogger("explorer.config")
class Config:
    def __init__(self, config_path: str):
        
        with open(config_path) as stream:
            try:
                self.yaml_dict: dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
      
        self.experiment_conf = ExperimentConfig(self.yaml_dict.get("ExperimentConfig", {}))
        self.connection_conf = ConnectionConfig(self.yaml_dict.get("ConnectionConfig", {}))
        self.model_conf = ModelConfig(self.yaml_dict.get("ModelConfig", {}))

    def dump_as_yaml(self, save_dir: str):
        """Creates a config.yaml file of the current config.

        Args:
            save_dir (str): The directory to save to.
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir / "config.yaml", 'w+') as ff:
            yaml.dump(yaml.safe_load(str({"ExperimentConfig" : vars(self.experiment_conf)})), stream=ff, default_flow_style=False)
            yaml.dump(yaml.safe_load(str({"ConnectionConfig" : vars(self.connection_conf)})), stream=ff, default_flow_style=False)
            yaml.dump(yaml.safe_load(str({"ModelConfig" : vars(self.model_conf)})), stream=ff, default_flow_style=False)

class ExperimentConfig:
    def __init__(self, yaml_dict: dict):
        #sets member variables to the values in yaml dict
        #set to default value, if yaml dict does not define a value
        self.host_processor: str = yaml_dict.get("host_processor", "cpu")
        self.max_search_trials: int = yaml_dict.get("max_search_trials", 6)
        self.top_k: int =  yaml_dict.get("top_k", 2)
        self.experiment_name: str = yaml_dict.get("experiment_name", f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}")
        self.nni_id: str = None

    def __setattr__(self, name, value):
        if name == "experiment_name":
            super().__setattr__("experiment_name", value)
            #If experiment name is set, pathes are adapted automatically
            self._experiment_dir: Path = MAIN_EXPERIMENT_DIR / self.experiment_name
            self._model_dir: Path  = self._experiment_dir / "models"
            self._metric_dir: Path = self._experiment_dir / "metrics"
            self._plot_dir: Path  = self._experiment_dir / "plots"
        else:
            super().__setattr__(name, value)

class ConnectionConfig:
    def __init__(self, yaml_dict: dict):
        try:
            self.target_name: str = yaml_dict["target_name"]
            self.target_user: str = yaml_dict["target_user"]
        except KeyError:
            logger.info("ConnectionConfig is not specified completely! Please specify or target connection is not possible.")
            exit(-1)

class ModelConfig:
    def __init__(self, yaml_dict: dict):
        self.model_type: str = yaml_dict["model_type"]
