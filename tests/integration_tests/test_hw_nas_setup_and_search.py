import os
from pathlib import Path
import shutil
import torch

import operator
from elasticai.explorer.hw_nas.constraints import ConstraintRegistry
from elasticai.explorer.hw_nas.estimators import (
    AccuracyEstimator,
    FLOPsEstimator,
    ParamEstimator,
)
from elasticai.explorer.hw_nas.hw_nas import (
    SearchAlgorithm,
)
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import DockerParams, RPICompiler
from elasticai.explorer.platforms.deployment.hw_manager import RPiHWManager
from elasticai.explorer.platforms.generator.generator import RPiGenerator
from elasticai.explorer.platforms.deployment.device_communication import (
    RPiHost,
    SSHParams,
)
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
        self.device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        accuracy_estimator = AccuracyEstimator(
            MLPTrainer, self.dataset_spec, 3, device=self.device
        )
        self.constraint_registry = ConstraintRegistry()
        self.constraint_registry.register_soft_constraint(
            estimator=accuracy_estimator, is_reward=True
        )

    def test_random_search(self):

        search_space = Path("elasticai/explorer/hw_nas/search_space/search_space.yaml")

        self.RPI5explorer.generate_search_space(search_space)
        top_k_models = self.RPI5explorer.search(
            constraint_registry=self.constraint_registry,
            search_algorithm=SearchAlgorithm.RANDOM_SEARCH,
        )
        assert len(top_k_models) == 2

    def test_grid_search(self):
        search_space = Path("elasticai/explorer/hw_nas/search_space/search_space.yaml")

        self.RPI5explorer.generate_search_space(search_space)
        top_k_models = self.RPI5explorer.search(
            constraint_registry=self.constraint_registry,
            search_algorithm=SearchAlgorithm.GRID_SEARCH,
        )
        assert len(top_k_models) == 2

    def test_evolution_search(self):
        search_space = Path("elasticai/explorer/hw_nas/search_space/search_space.yaml")
        self.RPI5explorer.generate_search_space(search_space)
        top_k_models = self.RPI5explorer.search(
            constraint_registry=self.constraint_registry,
            search_algorithm=SearchAlgorithm.EVOlUTIONARY_SEARCH,
        )
        assert len(top_k_models) == 2

    def test_constraint_search(self):
        search_space = Path("elasticai/explorer/hw_nas/search_space/search_space.yaml")
        self.RPI5explorer.generate_search_space(search_space)

        data_sample = torch.randn((1, 1, 28, 28), dtype=torch.float32, device=self.device)
        self.constraint_registry.register_hard_constraint(
            estimator=FLOPsEstimator(data_sample), operator=operator.lt, value=0
        )
        self.constraint_registry.register_hard_constraint(
            estimator=ParamEstimator(), operator=operator.lt, value=0
        )

        top_k_models = self.RPI5explorer.search(
            constraint_registry=self.constraint_registry,
            search_algorithm=SearchAlgorithm.EVOlUTIONARY_SEARCH,
        )
        assert len(top_k_models) == 0

    def test_generate_for_hw_platform(self):
        self.RPI5explorer.choose_target_hw(
            "rpi5",
            docker_params=DockerParams(),
            communication_params=SSHParams("", ""),
        )
        model = SampleMLP(28 * 28)

        self.RPI5explorer.generate_for_hw_platform(
            model=model, model_name=self.model_name, dataset_spec=self.dataset_spec
        )
        assert os.path.exists(self.RPI5explorer.model_dir / self.model_name) == True
        assert (
            type(torch.jit.load(self.RPI5explorer.model_dir / self.model_name))
            == torch.jit._script.RecursiveScriptModule  # type: ignore
        )

    def teardown_class(self):
        shutil.rmtree(self.RPI5explorer.experiment_dir, ignore_errors=True)
