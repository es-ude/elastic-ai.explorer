import datetime
import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from torch import nn

from elasticai.explorer import hw_nas, utils
from elasticai.explorer.config import DeploymentConfig, ModelConfig, HWNASConfig
from elasticai.explorer.knowledge_repository import KnowledgeRepository, HWPlatform
from elasticai.explorer.platforms.deployment.manager import HWManager
from elasticai.explorer.platforms.generator.generator import Generator
from elasticai.explorer.search_space import MLP
from settings import MAIN_EXPERIMENT_DIR


class Explorer:
    """
    The explorer class manages the HW-NAS and the deployment on hardware.
    It should be initialized with a KnowledgeRepository and config instances, to define the experiment setup.
    """

    def __init__(self, knowledge_repository: KnowledgeRepository, experiment_name: str = None):
        """
        Args:
            knowledge_repository
            experiment_name (str, optional): The name of the current experiment. Defaults to timestamp at instantiation.
              This defines in which directory the results are stored inside MAIN_EXPERIMENT_DIR (from settings.py).
        """
        self.logger = logging.getLogger("explorer")
        self.default_model: Optional[nn.Module] = None
        self.target_hw: Optional[HWPlatform] = None
        self.knowledge_repository = knowledge_repository
        self.generator = None
        self.hw_manager: Optional[HWManager] = None
        self.search_space = None
        self.hwnas_cfg = None
        self.deploy_cfg = None
        self.model_cfg = None

        if not experiment_name:
            self.experiment_name: str = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
        else:
            self.experiment_name: str = experiment_name

    @property
    def experiment_name(self):
        return self._experiment_name
    
    @property
    def experiment_dir(self):
        return self._experiment_dir
    
    @property
    def model_dir(self):
        return self._model_dir
    
    @property
    def metric_dir(self):
        return self._metric_dir
    
    @property
    def plot_dir(self):
        return self._plot_dir
    
    @experiment_name.setter
    def experiment_name(self, value):
        """Setting experiment name updates the experiment pathes aswell."""
        self._experiment_name = value
        self._experiment_dir: Path = MAIN_EXPERIMENT_DIR / self._experiment_name
        self._model_dir: Path  = self._experiment_dir / "models"
        self._metric_dir: Path = self._experiment_dir / "metrics"
        self._plot_dir: Path  = self._experiment_dir / "plots"
        self.logger.info(f"Experiment name: {self._experiment_name}")

    def set_default_model(self, model: nn.Module):
        self.default_model = model

    def set_model_cfg(self, model_cfg: ModelConfig):
        self.model_cfg = model_cfg
        self.model_cfg.dump_as_yaml(self._model_dir / "model_config.yaml")

    def generate_search_space(self):
        self.search_space = MLP()
        self.logger.info("Generated search space:\n %s", self.search_space)

    def choose_target_hw(self, name: str):
        self.target_hw: HWPlatform = self.knowledge_repository.fetch_hw_info(name)
        self.generator: Generator = self.target_hw.model_generator()
        self.hw_manager: HWManager = self.target_hw.platform_manager()
        self.logger.info("Configure chosen Target Hardware Platform. Name: %s, HW PLatform:\n%s", name, self.target_hw)

    def search(self, hwnas_cfg: HWNASConfig) -> list[any]:
        self.hwnas_cfg = hwnas_cfg
        self.logger.info("Start Hardware NAS with %d number of trials for top %d models ", 
                         self.hwnas_cfg.max_search_trials, self.hwnas_cfg.top_n_models)
        
        top_models, model_parameters, metrics = hw_nas.search(self.search_space, self.hwnas_cfg)

        utils.save_list_to_json(model_parameters, path_to_dir = self._model_dir, filename= "models.json")
        utils.save_list_to_json(metrics, path_to_dir = self._metric_dir, filename = "metrics.json")
        self.hwnas_cfg.dump_as_yaml(self._experiment_dir / "hwnas_config.yaml")

        return top_models

    def generate_for_hw_platform(self, model: Union[nn.Module, any], model_name: str) -> any:
        model_path = self._model_dir / model_name
        return self.generator.generate(model, model_path)

    def hw_setup_on_target(
            self, deploy_cfg: DeploymentConfig
    ):
        """Installs all necessary binaries and resources on the target platform

        Args:
            connection_conf (ConnectionConfig):
            
        """
        self.deploy_cfg = deploy_cfg
        self.logger.info("Setup Hardware target for experiments.")
        self.hw_manager.install_latency_measurement_on_target(self.deploy_cfg)
        self.hw_manager.install_accuracy_measurement_on_target(self.deploy_cfg, rebuild=False)
        self.deploy_cfg.dump_as_yaml(self._experiment_dir / "connection_config.yaml")

    def run_latency_measurement(
            self, model_name: str
    ) -> int:
        model_path = self._model_dir / model_name
        if self.deploy_cfg:
            self.hw_manager.deploy_model(self.deploy_cfg, model_path)
            return self.hw_manager.measure_latency(self.deploy_cfg, model_path)
        else:
            self.logger.error("Hardware was not setup on target before execution. Use Explorer.hw_setup_on_target() first!")
            exit(-1)

    def run_accuracy_measurement(
            self, model_name: str, path_to_data: Path
    ) -> float:
        model_path = self._model_dir / model_name
        if self.deploy_cfg:
            self.hw_manager.deploy_model(self.deploy_cfg, model_path)
            return self.hw_manager.measure_accuracy(
                self.deploy_cfg, model_path, path_to_data
            )
        else:
            self.logger.error("Hardware was not setup on target before execution. Use Explorer.hw_setup_on_target() first!")
            exit(-1)