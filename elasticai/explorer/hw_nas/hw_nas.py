from gc import callbacks
import logging
import math
from typing import Any, Callable, Type
from functools import partial
from enum import Enum

import optuna
from optuna.trial import FrozenTrial, TrialState
from optuna.study import MaxTrialsCallback
from torch.optim.adam import Adam

from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from elasticai.explorer.training import data
from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.hw_nas.cost_estimator import CostEstimator
from elasticai.explorer.training.trainer import Trainer

logger = logging.getLogger("explorer.nas")


class SearchAlgorithm(Enum):
    RANDOM_SEARCH = "random"
    GRID_SEARCH = "grid"
    EVOlUTIONARY_SEARCH = "evolution"


def objective_wrapper(
    trial: optuna.Trial,
    search_space_cfg: dict[str, Any],
    dataset_spec: data.DatasetSpecification,
    trainer_class: type[Trainer],
    device: str,
    n_estimation_epochs: int,
    flops_weight: float,
    constraints: dict[str, int] = {},
) -> float:

    def objective(trial: optuna.Trial) -> float:

        search_space = SearchSpace(search_space_cfg)
        model = search_space.create_model_sample(trial)
        optimizer = Adam(model.parameters(), lr=1e-3)
        trainer = trainer_class(device, optimizer, dataset_spec)

        cost_estimator = CostEstimator()
        sample, _ = next(iter(trainer.test_loader))
        flops = cost_estimator.estimate_flops(model, sample)
        params = cost_estimator.compute_num_params(model)
        max_flops = constraints.get("max_flops")
        max_params = constraints.get("max_params")
        if max_flops and flops > max_flops:
            logger.info(
                f"Trial {trial.number} pruned because flops {flops} > max_flops {max_flops}"
            )
            raise optuna.TrialPruned()
        if max_params and params > max_params:
            logger.info(
                f"Trial {trial.number} pruned because params {params} > max_params {max_params}"
            )
            raise optuna.TrialPruned()

        default = 0
        val_loss = float("inf")
        val_accuracy = 0
        flops_log10 = math.log10(flops)

        for epoch in range(n_estimation_epochs):
            trainer.train_epoch(model, epoch)
            val_accuracy, val_loss = trainer.validate(model)
            if val_accuracy:
                default = val_accuracy - (flops_log10 * flops_weight)
            else:
                val_accuracy = -1
                default = -val_loss - (flops_log10 * flops_weight)
            trial.report(default, epoch)
        trial.set_user_attr("val_accuracy", val_accuracy)
        trial.set_user_attr("flops_log10", flops_log10)
        trial.set_user_attr("val_loss", val_loss)
        return default

    return objective(trial)


def search(
    search_space_cfg: dict,
    hwnas_cfg: HWNASConfig,
    dataset_spec: data.DatasetSpecification,
    trainer_class: Type[Trainer],
) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    """
    Returns: top-models, model-parameters, metrics
    """
    search_space = SearchSpace(search_space_cfg)
    match hwnas_cfg.search_algorithm:
        case SearchAlgorithm.RANDOM_SEARCH:
            sampler = optuna.samplers.RandomSampler()
        case SearchAlgorithm.GRID_SEARCH:
            sampler = optuna.samplers.GridSampler(search_space_cfg)
        case SearchAlgorithm.EVOlUTIONARY_SEARCH:
            sampler = optuna.samplers.NSGAIISampler(
                population_size=20,
                mutation_prob=0.1,
            )
        case _:
            sampler = optuna.samplers.RandomSampler()

    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
    )

    if hwnas_cfg.count_only_completed_trials:
        n_trials = None
        callbacks = [MaxTrialsCallback(hwnas_cfg.max_search_trials, states=(TrialState.COMPLETE,))]
    else:
        n_trials = hwnas_cfg.max_search_trials
        callbacks = [MaxTrialsCallback(hwnas_cfg.max_search_trials, states=None)]

    study.optimize(
        partial(
            objective_wrapper,
            search_space_cfg=search_space_cfg,
            device=hwnas_cfg.host_processor,
            dataset_spec=dataset_spec,
            trainer_class=trainer_class,
            n_estimation_epochs=hwnas_cfg.n_estimation_epochs,
            flops_weight=hwnas_cfg.flops_weight,
            constraints=hwnas_cfg.hw_constraints,
        ),
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    test_results = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    eval: Callable[[FrozenTrial], float] = lambda trial: (
        trial.value if trial.value is not None else float("-inf")
    )
    test_results.sort(key=eval, reverse=True)

    top_k_frozen_trials = test_results[: hwnas_cfg.top_n_models]

    if len(top_k_frozen_trials) == 0:
        logger.warning("No models found in the search space.")
        return [], [], []

    top_k_models: list[Any] = []
    top_k_params: list[dict[str, Any]] = []
    top_k_metrics: list[dict] = []

    for frozen_trial in top_k_frozen_trials:
        top_k_models.append(search_space.create_model_sample(frozen_trial))
        top_k_params.append(frozen_trial.params)
        top_k_metrics.append(
            {
                "default": eval(frozen_trial),
                "val_accuracy": frozen_trial.user_attrs["val_accuracy"],
                "val_loss": frozen_trial.user_attrs["val_loss"],
                "flops log10": frozen_trial.user_attrs["flops_log10"],
            }
        )

    return top_k_models, top_k_params, top_k_metrics
