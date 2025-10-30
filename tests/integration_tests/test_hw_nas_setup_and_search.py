import os
from pathlib import Path
import pytest
import shutil
import torch
from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import RPICompiler
from elasticai.explorer.platforms.deployment.hw_manager import RPiHWManager
from elasticai.explorer.platforms.generator.generator import RPiGenerator
from elasticai.explorer.platforms.deployment.device_communication import RPiHost
from torchvision import transforms
from elasticai.explorer.training.trainer import SupervisedTrainer
from settings import ROOT_DIR
from tests.integration_tests.samples.sample_MLP import SampleMLP

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
                RPiGenerator,
                RPiHWManager,
                RPiHost,
                RPICompiler,
            )
        )
        self.RPI5explorer = Explorer(knowledge_repository)
        self.RPI5explorer.experiment_dir = Path(
            ROOT_DIR / "tests/integration_tests/test_experiment"
        )
        self.model_name = "ts_model_0.pt"
        self.hwnas_cfg = HWNASConfig(
            ROOT_DIR / Path("tests/integration_tests/test_configs/hwnas_config.yaml")
        )
        self.deploy_cfg = DeploymentConfig(
            ROOT_DIR
            / Path("tests/integration_tests/test_configs/deployment_config.yaml")
        )

        path_to_dataset = Path(ROOT_DIR / "data/mnist")
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dataset_spec = DatasetSpecification(
            dataset_type=MNISTWrapper,
            dataset_location=path_to_dataset,
            deployable_dataset_path=path_to_dataset,
            transform=transf,
        )

    @pytest.mark.parametrize(
        ("hwnas_cfg", "expected"),
        [
            ("tests/integration_tests/test_configs/hwnas_config.yaml", 2),
            ("tests/integration_tests/test_configs/grid_hwnas_config.yaml", 2),
            ("tests/integration_tests/test_configs/evolution_hwnas_config.yaml", 2),
            ("tests/integration_tests/test_configs/constraint_hwnas_config.yaml", 0),
        ],
    )
    def test_search(self, hwnas_cfg, expected):

        hwnas_cfg = HWNASConfig(config_path=ROOT_DIR / Path(hwnas_cfg))
        self.RPI5explorer.generate_search_space(
            Path(ROOT_DIR / "elasticai/explorer/hw_nas/search_space/search_space.yaml")
        )
        trainer = SupervisedTrainer(hwnas_cfg.host_processor, self.dataset_spec)
        top_k_models = self.RPI5explorer.search(hwnas_cfg, trainer)
        assert len(top_k_models) == expected

    def test_generate_for_hw_platform(self):
        self.RPI5explorer.choose_target_hw(self.deploy_cfg)
        model = SampleMLP(28 * 28)

        self.RPI5explorer.generate_for_hw_platform(
            model=model, model_name=self.model_name
        )
        assert os.path.exists(self.RPI5explorer.model_dir / self.model_name) == True
        assert (
            type(torch.jit.load(self.RPI5explorer.model_dir / self.model_name))
            == torch.jit._script.RecursiveScriptModule  # type: ignore
        )

    def teardown_class(self):
        shutil.rmtree(self.RPI5explorer.experiment_dir, ignore_errors=True)
