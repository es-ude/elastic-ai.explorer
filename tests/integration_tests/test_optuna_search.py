import pytest
from torch import optim

from elasticai.explorer.config import DeploymentConfig
from functools import partial
import optuna
import torch
from torchvision import transforms
from optuna.trial import TrialState

from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas import hw_nas
from elasticai.explorer.hw_nas.hw_nas import objective_wrapper
from elasticai.explorer.hw_nas.search_space.construct_search_space import (
    SearchSpace,
)
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.platforms.deployment.device_communication import Host
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.training.trainer import SupervisedTrainer
from settings import ROOT_DIR
from pathlib import Path
from types import SimpleNamespace

SAMPLE_PATH = ROOT_DIR / "tests/samples"
OUTPUT_PATH = ROOT_DIR / "tests/outputs"


@pytest.fixture(
    params=[
        {
            "flops_weight": 2,
            "n_estimation_epochs": 1,
            "max_search_trials": 8,
            "host_processor": "cpu",
            "top_n_models": 4,
        },
        {
            "flops_weight": 2,
            "n_estimation_epochs": 1,
            "max_search_trials": 2,
            "host_processor": "cpu",
            "top_n_models": 1,
        },
    ],
    ids=["8_trials_4_top_models", "2_trials_1_top_model"],
)
def hwnas_cfg(request):
    return SimpleNamespace(**request.param)


class TestFrozenTrialToModel:
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
            ROOT_DIR / "tests/integration_tests/test_experiment"
        )
        self.deploy_cfg = DeploymentConfig(
            Path(
                ROOT_DIR / "tests/integration_tests/test_configs/deployment_config.yaml"
            )
        )
        self.search_space_cfg = yaml_to_dict(
            Path(ROOT_DIR / "elasticai/explorer/hw_nas/search_space/search_space.yaml")
        )
        self.search_space = SearchSpace(self.search_space_cfg)
        self.dataset_spec = DatasetSpecification(
            MNISTWrapper,
            Path(ROOT_DIR / "data/mnist"),
            transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

    def test_frozentrial_to_model(self, hwnas_cfg):
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(),
            direction="maximize",
        )
        trainer = SupervisedTrainer(hwnas_cfg.host_processor, self.dataset_spec)
        study.optimize(
            partial(
                objective_wrapper,
                search_space_cfg=self.search_space_cfg,
                trainer_cls=trainer,
                n_estimation_epochs=hwnas_cfg.n_estimation_epochs,
                flops_weight=hwnas_cfg.flops_weight,
            ),
            n_trials=hwnas_cfg.max_search_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        test_results = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        assert len(test_results) == hwnas_cfg.max_search_trials

        for trial in test_results:
            model = self.search_space.create_model_sample(trial)
            input = torch.randn(1, 1, 28, 28).to(hwnas_cfg.host_processor)
            model.eval()
            model.to(hwnas_cfg.host_processor)
            result = model(input)
            assert len(result[0]) == 10

    def test_hw_nas_search(self, hwnas_cfg):
        trainer = SupervisedTrainer(hwnas_cfg.host_processor, self.dataset_spec)
        top_models, model_parameters, metrics = hw_nas.search(
            self.search_space_cfg,
            hwnas_cfg=hwnas_cfg,
            trainer=trainer,
        )
        assert len(top_models) == hwnas_cfg.top_n_models
        assert len(model_parameters) == hwnas_cfg.top_n_models
        assert len(metrics) == hwnas_cfg.top_n_models
        assert type(metrics[0]["val_accuracy"]) is float
