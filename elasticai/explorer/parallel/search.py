import copy
import logging
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
    _run_optuna_multiprocessing_search,
)
from elasticai.explorer.parallel.config import (
    CreateSamplerFn,
    MultiprocessingConfig,
    OptunaSearchConfig,
    SearchSpaceConfig,
)

_logger = logging.getLogger("explorer.nas")


def search_in_parallel(
    search_space_cfg: SearchSpaceConfig,
    create_sampler_fn: CreateSamplerFn,
    optimization_criteria: OptimizationCriteria,
    hw_nas_parameters: HWNASParameters,
    study_name: str,
    journal_file: str,
    devices: list[str],
    n_workers: int = 2,
    sampler_checkpoint_dir: Path | None = None,
) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    optimization_objective = partial(
        objective_on_device, criteria=optimization_criteria
    )
    optuna_search_config = OptunaSearchConfig(
        search_space_cfg=search_space_cfg,
        create_sampler_fn=create_sampler_fn,
        optimization_objective=optimization_objective,
        study_name=study_name,
        directions=("maximize",),
        max_search_trials=hw_nas_parameters.max_search_trials,
        count_only_completed_trials=hw_nas_parameters.count_only_completed_trials,
    )
    multiprocessing_config = MultiprocessingConfig(
        journal_file=journal_file,
        n_workers=n_workers,
        devices=devices,
        sampler_checkpoint_dir=sampler_checkpoint_dir,
    )

    study = run_optuna_search_in_parallel(
        optuna_search_config=optuna_search_config,
        multiprocessing_config=multiprocessing_config,
    )

    top_k_models, top_k_params, top_k_metrics = collect_top_k_results(
        study=study,
        hw_nas_parameters=hw_nas_parameters,
        optimization_criteria=optimization_criteria,
        search_space_cfg=search_space_cfg,
    )

    return top_k_models, top_k_params, top_k_metrics


def run_optuna_search_in_parallel(
    optuna_search_config: OptunaSearchConfig,
    multiprocessing_config: MultiprocessingConfig,
) -> optuna.Study:
    return _run_optuna_multiprocessing_search(
        optuna_search_config=optuna_search_config,
        multiprocessing_config=multiprocessing_config,
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
