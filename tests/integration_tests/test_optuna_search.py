import shutil
from functools import partial
import optuna
import torch
from torchvision import transforms
from optuna.trial import TrialState

from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas import hw_nas
from elasticai.explorer.hw_nas.hw_nas import (
    HWNASParameters,
    HardwareConstraints,
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
            flops_weight=2,
            n_estimation_epochs=1,
            max_search_trials=4,
            device="cpu",
            top_n_models=3,
        )
        self.hw_constraints = HardwareConstraints()
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
        self.dataset_spec = DatasetSpecification(
            dataset_type=MNISTWrapper,
            dataset_location=Path(ROOT_DIR / "data/mnist"),
            deployable_dataset_path=Path(ROOT_DIR / "data/mnist"),
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

    def test_frozentrial_to_model(self):
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(),
            direction="maximize",
        )
        study.optimize(
            partial(
                objective_wrapper,
                search_space_cfg=self.search_space_cfg,
                device=self.hw_nas_params.device,
                dataset_spec=self.dataset_spec,
                trainer_class=MLPTrainer,
                n_estimation_epochs=self.hw_nas_params.n_estimation_epochs,
                flops_weight=self.hw_nas_params.flops_weight,
                constraints=self.hw_constraints,
            ),
            n_trials=self.hw_nas_params.max_search_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        test_results = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        assert len(test_results) == self.hw_nas_params.max_search_trials

        for trial in test_results:
            model = self.search_space.create_model_sample(trial)
            input = torch.randn(1, 1, 28, 28).to(self.hw_nas_params.device)
            model.eval()
            model.to(self.hw_nas_params.device)
            result = model(input)
            assert len(result[0]) == 10

    def test_hw_nas_search(self):
        top_models, model_parameters, metrics = hw_nas.search(
            self.search_space_cfg,
            dataset_spec=self.dataset_spec,
            trainer_class=MLPTrainer,
            search_algorithm=SearchAlgorithm.RANDOM_SEARCH,
            hw_nas_parameters=self.hw_nas_params,
            hardware_constraints=self.hw_constraints,
        )
        assert len(top_models) == self.hw_nas_params.top_n_models
        assert len(model_parameters) == self.hw_nas_params.top_n_models
        assert len(metrics) == self.hw_nas_params.top_n_models
        assert type(metrics[0]["val_accuracy"]) is float

    def teardown_class(self):
        shutil.rmtree(self.RPI5explorer.experiment_dir, ignore_errors=True)
