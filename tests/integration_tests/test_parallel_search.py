from importlib.resources import as_file, files

import optuna
import pytest
from optuna.samplers import RandomSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from elasticai.explorer.hw_nas.estimators import ParamEstimator
from elasticai.explorer.hw_nas.hw_nas import HWNASParameters
from elasticai.explorer.hw_nas.optimization_criteria import OptimizationCriteria
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.parallel.config import (
    MultiprocessingConfig,
    OptunaSearchConfig,
)
from elasticai.explorer.parallel.search import (
    run_optuna_search_in_parallel,
    search_in_parallel,
)

SEARCH_SPACE_RESOURCE = files(__spec__.parent).joinpath("samples/search_space.yml")


@pytest.fixture
def search_space_config():
    with as_file(SEARCH_SPACE_RESOURCE) as f:
        return yaml_to_dict(f)


def make_sampler(worker_idx: int):
    return RandomSampler(seed=42 + worker_idx)


def standalone_objective(trial, search_space_cfg, device):
    x = trial.suggest_float("x", -1.0, 1.0)
    y = trial.suggest_float("y", -1.0, 1.0)
    return -(x**2 + y**2)  # maximize → optimum near (0, 0)


def test_optuna_parallel_search_completes(tmp_path):
    journal_file = tmp_path / "journal.log"
    optuna_search_config = OptunaSearchConfig(
        search_space_cfg={},
        create_sampler_fn=make_sampler,
        optimization_objective=standalone_objective,
        study_name="standalone",
        directions=("maximize",),
        max_search_trials=6,
    )
    multiprocessing_config = MultiprocessingConfig(
        journal_file=str(journal_file),
        n_workers=2,
        devices=["cpu", "cpu"],
    )
    study = run_optuna_search_in_parallel(
        optuna_search_config=optuna_search_config,
        multiprocessing_config=multiprocessing_config,
    )

    assert len(study.trials) >= 6
    pids = {t.user_attrs["worker_pid"] for t in study.trials}
    assert len(pids) == 2


def test_hw_nas_parallel_search_completes(
    tmp_path,
    search_space_config,
):
    criteria = OptimizationCriteria()
    criteria.register_objective(estimator=ParamEstimator())

    journal_file = tmp_path / "journal.log"

    search_in_parallel(
        search_space_cfg=search_space_config,
        create_sampler_fn=make_sampler,
        optimization_criteria=criteria,
        hw_nas_parameters=HWNASParameters(6, 3),
        study_name="test_parallel",
        journal_file=str(journal_file),
        n_workers=2,
        devices=["cpu", "cpu"],
    )

    study = optuna.load_study(
        study_name="test_parallel",
        storage=JournalStorage(JournalFileBackend(str(journal_file))),
    )
    pids = {t.user_attrs["worker_pid"] for t in study.trials}
    assert len(pids) == 2, f"Expected multiple workers, got PIDs: {pids}"


def test_hw_nas_parallel_search_with_sampler_checkpointing(
    tmp_path,
    search_space_config,
):
    criteria = OptimizationCriteria()
    criteria.register_objective(estimator=ParamEstimator())

    journal_file = tmp_path / "journal.log"
    checkpoint_dir = tmp_path / "checkpoints"

    # first run: produce sampler checkpoints
    search_in_parallel(
        search_space_cfg=search_space_config,
        create_sampler_fn=make_sampler,
        optimization_criteria=criteria,
        hw_nas_parameters=HWNASParameters(4, 2),
        study_name="test_parallel_ckpt",
        journal_file=str(journal_file),
        n_workers=2,
        devices=["cpu", "cpu"],
        sampler_checkpoint_dir=checkpoint_dir,
    )

    assert (checkpoint_dir / "sampler_worker_0.pkl").exists()
    assert (checkpoint_dir / "sampler_worker_1.pkl").exists()

    # second run with same study + same checkpoint dir: workers resume
    search_in_parallel(
        search_space_cfg=search_space_config,
        create_sampler_fn=make_sampler,
        optimization_criteria=criteria,
        hw_nas_parameters=HWNASParameters(8, 2),
        study_name="test_parallel_ckpt",
        journal_file=str(journal_file),
        n_workers=2,
        devices=["cpu", "cpu"],
        sampler_checkpoint_dir=checkpoint_dir,
    )

    study = optuna.load_study(
        study_name="test_parallel_ckpt",
        storage=JournalStorage(JournalFileBackend(str(journal_file))),
    )
    # second runs added trials to first run
    assert len(study.trials) >= 8
