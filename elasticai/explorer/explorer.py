from typing import Optional

from elasticai.explorer import hw_nas
from elasticai.explorer.knowledge_repository import KnowledgeRepository, HWPlatform

from elasticai.generator.generator import Generator


class Explorer:

    def __init__(self, knowledge_repository: KnowledgeRepository):
        self.target_hw: Optional[HWPlatform] = None
        self.knowledge_repository = knowledge_repository
        self.generator= None


    def choose_target_hw(self, name: str):
        self.target_hw: HWPlatform =self.knowledge_repository.fetch_hw_info(name)
        self.generator: Generator =self.target_hw.model_generator()

    def search(self):
        top_models=hw_nas.search()
        return top_models

    def generate_for_hw_platform(self, model, path):
        return self.generator.generate(model, path)










