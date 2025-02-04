import datetime
import logging

import yaml
from settings import MAIN_EXPERIMENT_DIR

logger = logging.getLogger("explorer.config")
class Config:
    def __init__(self, config_path: str):

        with open(config_path) as stream:
            try:
                yaml_dict: dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.experiment_conf = ExperimentConfig(yaml_dict.get("ExperimentConfig", {}))
        self.connection_conf = ConnectionConfig(yaml_dict.get("ConnectionConfig", {}))
        self.model_conf = ModelConfig(yaml_dict.get("ModelConfig", {}))


class ExperimentConfig:
    def __init__(self, yaml_dict: dict):
        
        #sets member variables to the values in yaml dict
        #set to default value, if yaml dict does not define a valu
        self.host_processor: str = yaml_dict.get("host_processor", "cpu")
        self.max_search_trials: int = yaml_dict.get("max_search_trials", 6)
        self.top_k: int =  yaml_dict.get("top_k", 2)
        self.experiment_name = yaml_dict.get("experiment_name", f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}")
        self.experiment_dir = MAIN_EXPERIMENT_DIR / self.experiment_name
        self.model_dir = self.experiment_dir / "models"
        self.metric_dir = self.experiment_dir / "metrics"
        self.plot_dir = self.experiment_dir / "plots"

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
