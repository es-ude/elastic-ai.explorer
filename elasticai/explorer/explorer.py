from typing import Optional

from torch import nn

from elasticai.explorer import hw_nas
from elasticai.explorer.knowledge_repository import KnowledgeRepository, HWPlatform
from elasticai.explorer.platforms.deployment.manager import HWManager, ConnectionData
from elasticai.explorer.platforms.generator.generator import Generator
from elasticai.explorer.search_space import MLP
import numpy as np


class Explorer:

    def __init__(self, knowledge_repository: KnowledgeRepository):
        self.default_model: Optional[nn.Module] = None
        self.target_hw: Optional[HWPlatform] = None
        self.knowledge_repository = knowledge_repository
        self.generator = None
        self.hw_manager: Optional[HWManager] = None
        self.search_space = None


    def set_default_model(self, model: nn.Module):
        self.default_model = model

    def generate_search_space(self):
        self.search_space = MLP()

    def choose_target_hw(self, name: str):
        self.target_hw: HWPlatform = self.knowledge_repository.fetch_hw_info(name)
        self.generator: Generator = self.target_hw.model_generator()
        self.hw_manager: HWManager = self.target_hw.platform_manager()

    def search(self, max_search_trials):
        top_models = hw_nas.search(self.search_space, max_search_trials)
        return top_models

    def generate_for_hw_platform(self, model, path):
        return self.generator.generate(model, path)
    
    def hw_setup_on_target(self,
        connection_info: ConnectionData,
    ):
        self.hw_manager.install_latency_measurement_on_target(connection_info)
        self.hw_manager.install_accuracy_measurement_on_target(connection_info)

    def run_latency_measurement(
        self,
        connection_info: ConnectionData,
        path_to_model,
        sample_size = 5
    ) -> int:
        
        self.hw_manager.deploy_model(connection_info, path_to_model)
        latencies = np.zeros(sample_size)

        for i in range(len(latencies)):
            latencies[i] = float(self.hw_manager.measure_latency(connection_info, path_to_model))

        return latencies.mean(), latencies.std()

    def run_accuracy_measurement(
        self,
        connection_info: ConnectionData,
        path_to_model,
        path_to_data
    ) -> int:
        
        self.hw_manager.deploy_model(connection_info, path_to_model)
        return self.hw_manager.measure_accuracy(connection_info, path_to_model, path_to_data)
