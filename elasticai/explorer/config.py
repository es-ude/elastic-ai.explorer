import datetime

import yaml
from settings import ROOT_DIR

class Config:
    def __init__(self, config_path: str):

        with open(config_path) as stream:
            try:
                yaml_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.experiment_conf = ExperimentConfig(yaml_dict["ExperimentConfig"])
        self.connection_conf = ConnectionConfig(yaml_dict["ConnectionConfig"])
        self.model_conf = ModelConfig(yaml_dict["ModelConfig"])


class ExperimentConfig:
    def __init__(self, yaml_dict: dict):
        self.host_device: str = yaml_dict["host_device"]
       
        self.max_search_trials: int = yaml_dict["max_search_trials"]
        self.top_k: int =  yaml_dict["top_k"]

        try:
            self.experiment_name = yaml_dict["experiment_name"]
        except KeyError:
            print("No experiment name given, timestamp is used as default.")
            self.experiment_name = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
        self.experiment_dir = ROOT_DIR / "experiments" / self.experiment_name


class ConnectionConfig:
    def __init__(self, yaml_dict: dict):
        self.target_host: str = yaml_dict["target_host"]
        self.target_user: str = yaml_dict["target_user"]

class ModelConfig:
    def __init__(self, yaml_dict: dict):
        self.model_type: str = yaml_dict["model_type"]
