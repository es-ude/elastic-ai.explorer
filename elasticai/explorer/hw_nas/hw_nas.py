from dataclasses import dataclass
import logging
import math
from typing import Any, Callable, Type
from functools import partial
from enum import Enum

import optuna
from optuna.trial import FrozenTrial, TrialState
from optuna.study import MaxTrialsCallback
import torch
from torch.optim.adam import Adam

from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from elasticai.explorer.training import data
from elasticai.explorer.hw_nas.cost_estimator import CostEstimator
from elasticai.explorer.training.trainer import Trainer

logger = logging.getLogger("explorer.nas")


@dataclass
class HardwareConstraints:
    max_flops: int | None = None
    max_params: int | None = None


@dataclass
class HWNASParameters:
    max_search_trials: int = 2
    top_n_models: int = 2
    n_estimation_epochs: int = 2
    flops_weight: float = 2.0
    count_only_completed_trials: bool = False
    device: str = "auto"


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
    constraints: HardwareConstraints,
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

        if constraints.max_flops and flops > constraints.max_flops:
            logger.info(
                f"Trial {trial.number} pruned because flops {flops} > max_flops {constraints.max_flops}"
            )
            raise optuna.TrialPruned()
        if constraints.max_params and params > constraints.max_params:
            logger.info(
                f"Trial {trial.number} pruned because params {params} > max_params {constraints.max_params}"
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
    dataset_spec: data.DatasetSpecification,
    trainer_class: Type[Trainer],
    search_algorithm: SearchAlgorithm,
    hardware_constraints: HardwareConstraints,
    hw_nas_parameters: HWNASParameters,
) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    """
    Returns: top-models, model-parameters, metrics
    """
    if hw_nas_parameters.device == "auto":
        hw_nas_parameters.device = "cuda" if torch.cuda.is_available() else "cpu"

    search_space = SearchSpace(search_space_cfg)

    match search_algorithm:
        case SearchAlgorithm.RANDOM_SEARCH:
            sampler = optuna.samplers.RandomSampler()
        case SearchAlgorithm.GRID_SEARCH:
            grid = search_space.to_grid()
            sampler = optuna.samplers.GridSampler(search_space=grid)
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

    if hw_nas_parameters.count_only_completed_trials:
        n_trials = None
        callbacks = [
            MaxTrialsCallback(
                hw_nas_parameters.max_search_trials, states=(TrialState.COMPLETE,)
            )
        ]
    else:
        n_trials = hw_nas_parameters.max_search_trials
        callbacks = [
            MaxTrialsCallback(hw_nas_parameters.max_search_trials, states=None)
        ]

    study.optimize(
        partial(
            objective_wrapper,
            search_space_cfg=search_space_cfg,
            device=hw_nas_parameters.device,
            dataset_spec=dataset_spec,
            trainer_class=trainer_class,
            n_estimation_epochs=hw_nas_parameters.n_estimation_epochs,
            flops_weight=hw_nas_parameters.flops_weight,
            constraints=hardware_constraints,
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

    top_k_frozen_trials = test_results[: hw_nas_parameters.top_n_models]

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
