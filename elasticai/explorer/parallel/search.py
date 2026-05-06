import copy
import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from elasticai.explorer.hw_nas.estimators import FLOPsEstimator, TrainMetricsEstimator
from elasticai.explorer.hw_nas.hw_nas import (
    HWNASParameters,
    collect_top_k_results,
    objective_wrapper,
)
from elasticai.explorer.hw_nas.optimization_criteria import (
    OptimizationCriteria,
)
from elasticai.explorer.parallel._multiprocessing_backend import (
    CreateSamplerFn,
    OptimizationObjective,
    SearchSpaceConfig,
    StudyDirection,
    _run_optuna_multiprocessing_search,
)

_logger = logging.getLogger("explorer.nas")


def search_in_parallel(
    search_space_cfg: SearchSpaceConfig,
    create_sampler_fn: CreateSamplerFn,
    optimization_criteria: OptimizationCriteria,
    hw_nas_parameters: HWNASParameters,
    study_name: str,
    journal_file: str,
    n_workers: int = 2,
    devices: list[str] | None = None,
    sampler_checkpoint_dir: Path | None = None,
) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    optimization_objective = partial(
        objective_on_device, criteria=optimization_criteria
    )

    study = run_optuna_search_in_parallel(
        search_space_cfg=search_space_cfg,
        create_sampler_fn=create_sampler_fn,
        study_name=study_name,
        journal_file=journal_file,
        optimization_objective=optimization_objective,
        direction="maximize",
        n_workers=n_workers,
        devices=devices,
        max_search_trials=hw_nas_parameters.max_search_trials,
        count_only_completed_trials=hw_nas_parameters.count_only_completed_trials,
        sampler_checkpoint_dir=sampler_checkpoint_dir,
    )

    top_k_models, top_k_params, top_k_metrics = collect_top_k_results(
        study=study,
        hw_nas_parameters=hw_nas_parameters,
        optimization_criteria=optimization_criteria,
        search_space_cfg=search_space_cfg,
    )

    return top_k_models, top_k_params, top_k_metrics


def run_optuna_search_in_parallel(
    search_space_cfg: SearchSpaceConfig,
    create_sampler_fn: CreateSamplerFn,
    optimization_objective: OptimizationObjective,
    study_name: str,
    journal_file: str,
    direction: StudyDirection | None = None,
    directions: Sequence[StudyDirection] | None = None,
    n_workers: int = 1,
    devices: list[str] | None = None,
    max_search_trials: int | None = None,
    count_only_completed_trials: bool = False,
    sampler_checkpoint_dir: Path | None = None,
) -> optuna.Study:
    return _run_optuna_multiprocessing_search(
        search_space_cfg=search_space_cfg,
        create_sampler_fn=create_sampler_fn,
        optimization_objective=optimization_objective,
        study_name=study_name,
        journal_file=journal_file,
        direction=direction,
        directions=directions,
        n_workers=n_workers,
        devices=devices,
        max_search_trials=max_search_trials,
        count_only_completed_trials=count_only_completed_trials,
        sampler_checkpoint_dir=sampler_checkpoint_dir,
    )


def is_duplicated_trial(trial: optuna.Trial) -> bool:
    states_to_consider = (
        TrialState.RUNNING,
        TrialState.COMPLETE,
    )
    trials_to_consider = trial.study.get_trials(
        deepcopy=False, states=states_to_consider
    )

    for t in trials_to_consider:
        if t.number == trial.number:
            continue
        if t.params == trial.params:
            return True

    return False


def objective_on_device(trial, search_space_cfg, device, criteria):
    local_criteria = _bind_criteria_to_device(criteria, device)
    return objective_wrapper(trial, search_space_cfg, local_criteria)


def _bind_criteria_to_device(
    criteria: OptimizationCriteria,
    device: str,
) -> OptimizationCriteria:
    # bind criteria to the local worker
    local_criteria = copy.deepcopy(criteria)

    for estimator in local_criteria.get_estimators():
        if isinstance(estimator, FLOPsEstimator):
            estimator.data_sample = estimator.data_sample.to(device)

        if isinstance(estimator, TrainMetricsEstimator):
            estimator.trainer.device = device

    return local_criteria
