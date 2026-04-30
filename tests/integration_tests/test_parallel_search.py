import optuna
from optuna.samplers import RandomSampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from elasticai.explorer.hw_nas.estimators import ParamEstimator
from elasticai.explorer.hw_nas.hw_nas import HWNASParameters
from elasticai.explorer.hw_nas.optimization_criteria import OptimizationCriteria
from elasticai.explorer.hw_nas.search_space.utils import yaml_to_dict
from elasticai.explorer.parallel.search import search_in_parallel
from settings import ROOT_DIR

SEARCH_SPACE_PATH = ROOT_DIR / "tests/integration_tests/samples/search_space.yml"


def make_sampler(worker_idx: int):
    return RandomSampler(seed=42 + worker_idx)


def test_multiprocessing_search_completes(tmp_path):
    search_space_cfg = yaml_to_dict(SEARCH_SPACE_PATH)
    criteria = OptimizationCriteria()
    criteria.register_objective(estimator=ParamEstimator())

    journal_file = tmp_path / "journal.log"

    search_in_parallel(
        search_space_cfg=search_space_cfg,
        sampler_builder=make_sampler,
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
    print(pids)
    assert len(pids) >= 2, f"Expected multiple workers, got PIDs: {pids}"


def test_multiprocessing_search_with_sampler_checkpointing(tmp_path):
    search_space_cfg = yaml_to_dict(SEARCH_SPACE_PATH)
    criteria = OptimizationCriteria()
    criteria.register_objective(estimator=ParamEstimator())

    journal_file = tmp_path / "journal.log"
    checkpoint_dir = tmp_path / "checkpoints"

    # first run: produce sampler checkpoints
    search_in_parallel(
        search_space_cfg=search_space_cfg,
        sampler_builder=make_sampler,
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
        search_space_cfg=search_space_cfg,
        sampler_builder=make_sampler,
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
