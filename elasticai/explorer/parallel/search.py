import copy
import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable

from optuna.samplers import BaseSampler

from elasticai.explorer.hw_nas.estimators import FLOPsEstimator, TrainMetricsEstimator
from elasticai.explorer.hw_nas.hw_nas import (
    HWNASParameters,
    collect_top_k_results,
    objective_wrapper,
)
from elasticai.explorer.hw_nas.optimization_criteria import (
    OptimizationCriteria,
)
from elasticai.explorer.parallel.optuna_runner import (
    run_parallel_optuna_search,
)

logger = logging.getLogger("explorer.nas")


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


def parallelized_objective_wrapper(trial, search_space_cfg, device, criteria):
    local_criteria = _bind_criteria_to_device(criteria, device)
    return objective_wrapper(trial, search_space_cfg, local_criteria)


def search_in_parallel(
    search_space_cfg: dict,
    sampler_builder: Callable[[int], BaseSampler],
    optimization_criteria: OptimizationCriteria,
    hw_nas_parameters: HWNASParameters,
    study_name: str,
    journal_file: str,
    n_workers: int = 2,
    devices: list[str] | None = None,
    sampler_checkpoint_dir: Path | None = None,
) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    optimization_objective = partial(
        parallelized_objective_wrapper, criteria=optimization_criteria
    )

    study = run_parallel_optuna_search(
        search_space_cfg=search_space_cfg,
        sampler_builder=sampler_builder,
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
