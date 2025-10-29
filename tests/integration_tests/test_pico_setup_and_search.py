from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer

from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.knowledge_repository import KnowledgeRepository
from elasticai.explorer.generator.deployment.compiler import PicoCompiler
from elasticai.explorer.generator.model_compiler.model_compiler import (
    TFliteModelCompiler,
)
from elasticai.explorer.generator.deployment.device_communication import PicoHost
from elasticai.explorer.generator.deployment.hw_manager import PicoHWManager
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.training.trainer import MLPTrainer
from settings import ROOT_DIR
from torchvision import transforms

from pathlib import Path

SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"


class TestPicoHWNasSetupAndSearch:
    """Integration test of the Explorer HW-NAS pipeline without a target device."""

    def setup_method(self):
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            Generator(
                "pico",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                TFliteModelCompiler,
                PicoHWManager,
                PicoHost,
                PicoCompiler,
            )
        )
        self.explorer = Explorer(knowledge_repository)
        self.explorer.experiment_dir = Path("tests/integration_tests/test_experiment")

        self.hwnas_cfg = HWNASConfig(
            config_path=Path("tests/integration_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            config_path=Path(
                "tests/integration_tests/test_configs/deployment_config.yaml"
            )
        )
        path_to_dataset = Path(ROOT_DIR / "data/mnist")
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dataset_spec = DatasetSpecification(
            dataset_type=MNISTWrapper,
            dataset_location=path_to_dataset,
            transform=transf,
        )

    def test_search(self):
        self.explorer.generate_search_space(
            ROOT_DIR / Path("elasticai/explorer/hw_nas/search_space/search_space.yaml")
        )
        top_k_models = self.explorer.search(
            self.hwnas_cfg, self.dataset_spec, MLPTrainer
        )
        assert len(top_k_models) == 2
