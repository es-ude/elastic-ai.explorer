import os
from elasticai.explorer import search_space
from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas.search_space.construct_sp import CombinedSearchSpace, yml_to_dict
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import PicoCompiler
from elasticai.explorer.platforms.generator.generator import PicoGenerator
from elasticai.explorer.platforms.deployment.device_communication import PicoHost, RPiHost
from elasticai.explorer.platforms.deployment.manager import PicoHWManager
from settings import ROOT_DIR
from tests.integration_tests.samples.sample_MLP import sample_MLP
from pathlib import Path

SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"


class TestPicoHWNasSetupAndSearch:
    """Integration test of the Explorer HW-NAS pipeline without a target device."""

    def setup_method(self):
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            HWPlatform(
                "pico",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                PicoGenerator,
                PicoHWManager,
                PicoHost,
                PicoCompiler,
            )
        )
        self.explorer = Explorer(knowledge_repository)
        self.explorer.experiment_dir = Path(
            "tests/integration_tests/test_experiment"
        )
        self.model_name = "model"

        self.hwnas_cfg = HWNASConfig(
            config_path=Path("tests/integration_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            config_path=Path(
                "tests/integration_tests/test_configs/deployment_config.yaml"
            )
        )

    def test_search(self):
        search_space = yml_to_dict(
        Path("tests/integration_tests/samples/search_space.yml")
    )
        search_space = CombinedSearchSpace(search_space)
        self.explorer.generate_search_space(search_space)
        top_k_models = self.explorer.search(self.hwnas_cfg)
        assert len(top_k_models) == 1

    def teardown_method(self):
        os.remove(
            self.explorer.model_dir / (self.model_name + ".tflite"),
        )
        os.remove(self.explorer.model_dir / (self.model_name + ".cpp"))
