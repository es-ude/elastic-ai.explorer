from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.explorer import Explorer

from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import PicoCompiler
from elasticai.explorer.platforms.generator.generator import PicoGenerator
from elasticai.explorer.platforms.deployment.device_communication import PicoHost
from elasticai.explorer.platforms.deployment.hw_manager import PicoHWManager
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
        self.dataset_spec = DatasetSpecification(MNISTWrapper, path_to_dataset, None, transf)

    def test_search(self):
        self.explorer.generate_search_space(
            ROOT_DIR / Path("elasticai/explorer/hw_nas/search_space/search_space.yaml")
        )
        top_k_models = self.explorer.search(
            self.hwnas_cfg, self.dataset_spec, MLPTrainer
        )
        assert len(top_k_models) == 2
