import os
import torch
from elasticai.explorer.hw_nas import search_space
from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.platforms.deployment.device_communication import Host
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from settings import ROOT_DIR
from tests.integration_tests.samples.sample_MLP import sample_MLP
from pathlib import Path

SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"


class TestHWNasSetupAndSearch:
    """Integration test of the Explorer HW-NAS pipeline without a target device."""

    def setUp(self):
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            HWPlatform(
                "rpi5",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                PIGenerator,
                PIHWManager,
                Host,
                Compiler,
            )
        )
        self.RPI5explorer = Explorer(knowledge_repository)
        self.RPI5explorer.experiment_dir = Path(
            "tests/integration_tests/test_experiment"
        )
        self.model_name = "ts_model_0.pt"
        self.hwnas_cfg = HWNASConfig()
        self.deploy_cfg = DeploymentConfig(
            Path("tests/integration_tests/test_configs/deployment_config.yaml")
        )

    def test_search(self):
        self.setUp()
        self.RPI5explorer.generate_search_space()
        top_k_models = self.RPI5explorer.search(self.hwnas_cfg)
        assert len(top_k_models) == 2

    def test_generate_for_hw_platform(self):
        self.setUp()
        self.RPI5explorer.choose_target_hw(self.deploy_cfg)
        model = sample_MLP()

        self.RPI5explorer.generate_for_hw_platform(
            model=model, model_name=self.model_name
        )
        assert os.path.exists(self.RPI5explorer.model_dir / self.model_name) == True
        assert (
            type(torch.jit.load(self.RPI5explorer.model_dir / self.model_name))
            == torch.jit._script.RecursiveScriptModule  # type: ignore
        )
