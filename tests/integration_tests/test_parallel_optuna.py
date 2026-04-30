import pytest
import optuna
from optuna.samplers import RandomSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from elasticai.explorer.parallel.optuna_runner import run_parallel_optuna_search


def make_sampler(worker_idx: int):
    return RandomSampler(seed=42 + worker_idx)


def standalone_objective(trial, search_space_cfg, device):
    x = trial.suggest_float("x", -1.0, 1.0)
    y = trial.suggest_float("y", -1.0, 1.0)
    return -(x**2 + y**2)  # maximize → optimum near (0, 0)


def test_run_parallel_optuna_nas_minimal(tmp_path):
    journal_file = tmp_path / "journal.log"
    study = run_parallel_optuna_search(
        search_space_cfg={},
        sampler_builder=make_sampler,
        optimization_objective=standalone_objective,
        study_name="standalone",
        journal_file=str(journal_file),
        direction="maximize",
        n_workers=2,
        devices=["cpu", "cpu"],
        max_search_trials=6,
    )

    assert len(study.trials) >= 6
    pids = {t.user_attrs["worker_pid"] for t in study.trials}
    assert len(pids) >= 2


def test_raises_when_no_direction(tmp_path):
    with pytest.raises(ValueError, match="Either direction or directions"):
        run_parallel_optuna_search(
            search_space_cfg={},
            sampler_builder=make_sampler,
            optimization_objective=standalone_objective,
            study_name="x",
            journal_file=str(tmp_path / "j.log"),
            n_workers=2,
            devices=["cpu"],
        )


def test_raises_when_both_directions_set(tmp_path):
    with pytest.raises(ValueError, match="Greedy"):
        run_parallel_optuna_search(
            search_space_cfg={},
            sampler_builder=make_sampler,
            optimization_objective=standalone_objective,
            study_name="x",
            journal_file=str(tmp_path / "j.log"),
            n_workers=2,
            devices=["cpu"],
            direction="maximize",
            directions=["maximize"],
        )
