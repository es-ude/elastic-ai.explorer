from typing import Optional

from torch import nn

from elasticai.explorer import hw_nas
from elasticai.explorer.knowledge_repository import KnowledgeRepository, HWPlatform
from elasticai.explorer.platforms.deployment.manager import HWManager
from elasticai.explorer.platforms.generator.generator import Generator
from elasticai.explorer.search_space import MLP



class Explorer:

    def __init__(self, knowledge_repository: KnowledgeRepository):
        self.default_model: Optional[nn.Module] = None
        self.target_hw: Optional[HWPlatform] = None
        self.knowledge_repository = knowledge_repository
        self.generator= None
        self.hw_manager: Optional[HWManager]=None
        self.search_space=None


    def set_default_model(self, model:nn.Module):
        self.default_model=model

    def generate_search_space(self):
       self.search_space=MLP()

    def choose_target_hw(self, name: str):
        self.target_hw: HWPlatform =self.knowledge_repository.fetch_hw_info(name)
        self.generator: Generator =self.target_hw.model_generator()
        self.hw_manager: HWManager= self.target_hw.platform_manager()

    def search(self):
        top_models=hw_nas.search(self.search_space)
        return top_models

    def generate_for_hw_platform(self, model, path):
        return self.generator.generate(model, path)


    def run_measurement(self, path_to_model, )->int:
       return self.hw_manager.deploy_model(path_to_model)










