import os
from pathlib import Path
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from elasticai.explorer import search_space
from elasticai.explorer import explorer
from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.data import DatasetInfo
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.platforms.deployment.device_communication import Host
from elasticai.explorer.platforms.deployment.manager import PIHWManager

from settings import ROOT_DIR
from tests.integration_tests.samples.sample_MLP import sample_MLP


SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"


class TestHWNasSetupAndSearch:
    """Integration test of the Explorer HW-NAS pipeline without a target device."""

    def setup_class(self):
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
        self.hwnas_cfg = HWNASConfig(
            config_path=Path("tests/integration_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            config_path=Path(
                "tests/integration_tests/test_configs/deployment_config.yaml"
            )
        )

        path_to_dataset = Path("data/mnist")
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        __test_dataset = MNIST(path_to_dataset, download=True, transform=transf)
        self.dataset_info = DatasetInfo(MNIST,path_to_dataset, transf)

    def test_search(self):
        self.RPI5explorer.generate_search_space()
        top_k_models = self.RPI5explorer.search(self.hwnas_cfg, self.dataset_info)
        assert len(top_k_models) == 1
        assert type(top_k_models[0]) == search_space.MLP

    def test_generate_for_hw_platform(self):
        self.RPI5explorer.choose_target_hw(self.deploy_cfg)
        model = sample_MLP()

        self.RPI5explorer.generate_for_hw_platform(
            model=model, model_name=self.model_name
        )
        assert (
            os.path.exists(
                self.RPI5explorer.model_dir / self.model_name
            )
            == True
        )
        assert (
            type(
                torch.jit.load(
                    self.RPI5explorer.model_dir /  self.model_name
                )
            )
            == torch.jit._script.RecursiveScriptModule  # type: ignore
        )
