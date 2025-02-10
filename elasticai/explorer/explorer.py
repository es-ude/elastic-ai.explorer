import logging
import os
from typing import Optional

import numpy as np
from torch import nn

from elasticai.explorer import hw_nas
from elasticai.explorer.config import Config
from elasticai.explorer.knowledge_repository import KnowledgeRepository, HWPlatform
from elasticai.explorer.platforms.deployment.manager import HWManager
from elasticai.explorer.platforms.generator.generator import Generator
from elasticai.explorer.search_space import MLP


class Explorer:

    def __init__(self, knowledge_repository: KnowledgeRepository, config: Config):
        """Initializes Explorer instance.

        Args:
            knowledge_repository (KnowledgeRepository): Gives information on the target platform.
            config (Config): Consist of experiment_conf, connection_conf and model_conf
        """
        self.logger = logging.getLogger("explorer")
        self.default_model: Optional[nn.Module] = None
        self.target_hw: Optional[HWPlatform] = None
        self.knowledge_repository = knowledge_repository
        self.generator = None
        self.hw_manager: Optional[HWManager] = None
        self.search_space = None
        self.config = config
        
        #shortcut to the individual configs
        self.experiment_conf = config.experiment_conf
        self.connection_conf = config.connection_conf
        self.model_conf = config.model_conf
        

    def set_default_model(self, model: nn.Module):
        self.default_model = model

    def generate_search_space(self):
        self.search_space = MLP()
        self.logger.info("Generated search space:\n %s", self.search_space)

    def choose_target_hw(self, name: str):
        self.target_hw: HWPlatform = self.knowledge_repository.fetch_hw_info(name)
        self.generator: Generator = self.target_hw.model_generator()
        self.hw_manager: HWManager = self.target_hw.platform_manager()
        self.logger.info("Configure chosen Target Hardware Platform. Name: %s, HW PLatform:\n%s", name, self.target_hw)

    def search(self) -> list[any]:
        self.logger.info("Start Hardware NAS with %d number of trials for top %d models ", 
                         self.experiment_conf.max_search_trials, self.experiment_conf.top_n_models)
        top_models = hw_nas.search(self.search_space, self.experiment_conf)
        #save actual config with all defaults
        self.config.dump_as_yaml(self.experiment_conf._experiment_dir)
        return top_models

    def generate_for_hw_platform(self, model, path) -> any:
        return self.generator.generate(model, path)

    def hw_setup_on_target(
            self,
            
    ):
        self.logger.info("Setup Hardware target for experiments.")
        self.hw_manager.install_latency_measurement_on_target(self.connection_conf)
        self.hw_manager.install_accuracy_measurement_on_target(self.connection_conf)

    def run_latency_measurement(
            self, path_to_model: str
    ) -> int:
        self.hw_manager.deploy_model(self.connection_conf, path_to_model)
        return self.hw_manager.measure_latency(self.connection_conf, path_to_model)

    def run_accuracy_measurement(
            self,  path_to_model: str, path_to_data: str
    ) -> float:
        self.hw_manager.deploy_model(self.connection_conf, path_to_model)
        return self.hw_manager.measure_accuracy(
            self.connection_conf, path_to_model, path_to_data
        )