import os
from pathlib import Path
import pytest
import torch
import yaml
from elasticai.explorer.hw_nas import search_space
from elasticai.explorer.config import HWNASConfig, DeploymentConfig
from elasticai.explorer.hw_nas.search_space.construct_sp import CombinedSearchSpace
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.platforms.deployment.device_communication import Host
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from torchvision import transforms
from elasticai.explorer.training.trainer import MLPTrainer
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

        path_to_dataset = Path("data/mnist")
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dataset_spec = DatasetSpecification(MNISTWrapper, path_to_dataset, transf)

        path_to_dataset = Path("data/mnist")
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dataset_spec = DatasetSpecification(MNISTWrapper, path_to_dataset, transf)

    @pytest.fixture
    def search_space_dict(self):
        return yaml.safe_load(
            """input: [1, 28, 28]
output: 10
blocks:
  - block:  "1" #namefield muss noch rein
    op_candidates: ["linear", "conv2d"]
    depth: [1, 2, 3]
    linear:
      #überall range oder choices
   #   activation: [ "relu", "sigmoid"]
      width: [16, 32, 5, 4]
    conv2D:
      kernel_size: [1, 2]
      stride: [1, 2]
      out_channels: [ 10, 4]"""
        )

    def test_search(self, search_space_dict):
        self.RPI5explorer.generate_search_space(CombinedSearchSpace(search_space_dict))  # type: ignore
        top_k_models = self.RPI5explorer.search(
            self.hwnas_cfg, self.dataset_spec, MLPTrainer
        )
        assert len(top_k_models) == 2

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
