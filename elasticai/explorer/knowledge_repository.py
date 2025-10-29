from elasticai.explorer.generator.generator import Generator


class KnowledgeRepository:
    def __init__(self):
        self.supported_hw_platforms = {}

    def register_hw_platform(self, platform: Generator):
        self.supported_hw_platforms[platform.hw_platform_name] = platform

    def fetch_hw_info(self, name: str) -> Generator:
        return self.supported_hw_platforms[name]
