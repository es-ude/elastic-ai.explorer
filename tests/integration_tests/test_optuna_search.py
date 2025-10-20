import shutil
from functools import partial
import optuna
import torch
from torchvision import transforms
from optuna.trial import TrialState

from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas import hw_nas
from elasticai.explorer.hw_nas.constraints import ConstraintRegistry
from elasticai.explorer.hw_nas.estimators import AccuracyEstimator
from elasticai.explorer.hw_nas.hw_nas import (
    HWNASParameters,
    SearchAlgorithm,
    objective_wrapper,
)
from elasticai.explorer.hw_nas.search_space.construct_search_space import (
    SearchSpace,
)
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.generator.generator import RPiGenerator
from elasticai.explorer.platforms.deployment.device_communication import Host
from elasticai.explorer.platforms.deployment.hw_manager import RPiHWManager
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.training.trainer import MLPTrainer
from settings import ROOT_DIR
from pathlib import Path

SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"


class TestFrozenTrialToModel:
    """Integration test of the Explorer HW-NAS pipeline without a target device."""

    def setup_class(self):
        self.hw_nas_params = HWNASParameters(
            max_search_trials=3,
            top_n_models=3,
        )

        self.dataset_spec = DatasetSpecification(
            dataset_type=MNISTWrapper,
            dataset_location=Path(ROOT_DIR / "data/mnist"),
            deployable_dataset_path=Path(ROOT_DIR / "data/mnist"),
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        self.device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        accuracy_estimator = AccuracyEstimator(
            MLPTrainer, self.dataset_spec, 2, device=self.device
        )
        self.constraint_registry = ConstraintRegistry()
        self.constraint_registry.register_soft_constraint(
            estimator=accuracy_estimator, is_reward=True
        )
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            HWPlatform(
                "rpi5",
                "Raspberry PI 5 with A76 processor and 8GB RAM",
                RPiGenerator,
                RPiHWManager,
                Host,
                Compiler,
            )
        )
        self.RPI5explorer = Explorer(knowledge_repository)
        self.RPI5explorer.experiment_dir = Path(
            ROOT_DIR / "tests/integration_tests/test_experiment"
        )

        self.search_space_cfg = yaml_to_dict(
            Path(ROOT_DIR / "elasticai/explorer/hw_nas/search_space/search_space.yaml")
        )
        self.search_space = SearchSpace(self.search_space_cfg)

    def test_frozentrial_to_model(self):
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(),
            direction="maximize",
        )
        study.optimize(
            partial(
                objective_wrapper,
                search_space_cfg=self.search_space_cfg,
                constraint_registry=self.constraint_registry,
            ),
            n_trials=self.hw_nas_params.max_search_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        test_results = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        assert len(test_results) == self.hw_nas_params.max_search_trials

        for trial in test_results:
            model = self.search_space.create_model_sample(trial)
            input = torch.randn(1, 1, 28, 28).to(self.device)
            model.eval()
            model.to(self.device)
            result = model(input)
            assert len(result[0]) == 10

    def test_hw_nas_search(self):
        top_models, model_parameters, metrics = hw_nas.search(
            self.search_space_cfg,
            search_algorithm=SearchAlgorithm.RANDOM_SEARCH,
            constraint_registry=self.constraint_registry,
            hw_nas_parameters=self.hw_nas_params,
        )
        assert len(top_models) == self.hw_nas_params.top_n_models
        assert len(model_parameters) == self.hw_nas_params.top_n_models
        assert len(metrics) == self.hw_nas_params.top_n_models
        assert type(metrics[0]["accuracy_estimate"]) is float

    def teardown_class(self):
        shutil.rmtree(self.RPI5explorer.experiment_dir, ignore_errors=True)
