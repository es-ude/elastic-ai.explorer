import os
from pathlib import Path
import pytest
import shutil
import torch

import operator
from elasticai.explorer.hw_nas.optimization_criteria import OptimizationCriteria
from elasticai.explorer.hw_nas.estimators import (
    FLOPsEstimator,
    ParamEstimator,
    TrainMetricsEstimator,
)
from elasticai.explorer.hw_nas.hw_nas import (
    HWNASParameters,
    SearchStrategy,
)
from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import KnowledgeRepository
from elasticai.explorer.generator.deployment.compiler import DockerParams, RPICompiler
from elasticai.explorer.generator.deployment.hw_manager import RPiHWManager
from elasticai.explorer.generator.model_compiler.model_compiler import (
    TorchscriptModelCompiler,
)
from elasticai.explorer.generator.deployment.device_communication import (
    RPiHost,
    SSHParams,
)
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
            Generator(
                "rpi5",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                TorchscriptModelCompiler,
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
        trainer = SupervisedTrainer(
            self.device,
            self.dataset_spec,
            batch_size=64,
        )

        accuracy_estimator = TrainMetricsEstimator(
            trainer, metric_name="accuracy", n_estimation_epochs=1
        )
        self.optimization_criteria = OptimizationCriteria()
        self.optimization_criteria.register_objective(estimator=accuracy_estimator)

        self.search_space = self.RPI5explorer.generate_search_space(
            ROOT_DIR / Path("tests/integration_tests/samples/search_space.yml")
        )

    @pytest.mark.parametrize(
        ("search_strategy", "with_hardconstraints", "expected"),
        [
            (SearchStrategy.RANDOM_SEARCH, False, 2),
            (SearchStrategy.EVOLUTIONARY_SEARCH, False, 2),
            (SearchStrategy.RANDOM_SEARCH, True, 0),
        ],
    )
    def test_search(self, search_strategy, with_hardconstraints, expected):
        if with_hardconstraints:
            data_sample = torch.randn(
                (1, 1, 28, 28), dtype=torch.float32, device=self.device
            )
            self.optimization_criteria.register_hard_constraint(
                estimator=FLOPsEstimator(data_sample), operator=operator.lt, value=0
            )
            self.optimization_criteria.register_hard_constraint(
                estimator=ParamEstimator(), operator=operator.lt, value=0
            )
        top_k_models, _ = self.RPI5explorer.search(
            optimization_criteria=self.optimization_criteria,
            search_strategy=search_strategy,
            hw_nas_parameters=HWNASParameters(2, 2),
        )
        assert len(top_k_models) == expected

    def test_generate_for_hw_platform(self):
        self.RPI5explorer.choose_target_hw(
            "rpi5",
            compiler_params=DockerParams(),
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
