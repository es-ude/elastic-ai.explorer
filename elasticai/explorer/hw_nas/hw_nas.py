from dataclasses import dataclass
import logging
from typing import Any, Callable
from functools import partial
from enum import Enum

import optuna
from optuna.trial import FrozenTrial, TrialState
from optuna.study import MaxTrialsCallback

from elasticai.explorer.hw_nas.constraints import ConstraintRegistry
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace

logger = logging.getLogger("explorer.nas")


@dataclass
class HWNASParameters:
    max_search_trials: int = 2
    top_n_models: int = 2
    count_only_completed_trials: bool = False


class SearchAlgorithm(Enum):
    RANDOM_SEARCH = "random"
    GRID_SEARCH = "grid"
    EVOlUTIONARY_SEARCH = "evolution"


def objective_wrapper(
    trial: optuna.Trial,
    search_space_cfg: dict[str, Any],
    constraint_registry: ConstraintRegistry,
) -> float:

    def objective(trial: optuna.Trial) -> float:

        search_space = SearchSpace(search_space_cfg)
        model = search_space.create_model_sample(trial)

        soft_score = 0.0
        for estimator in constraint_registry:

            estimate = estimator.estimate(model)
            trial.set_user_attr(estimator.metric_id, estimate)
            hard_constraints = constraint_registry.get_hard_constraints(estimator)
            for hc in hard_constraints:
                if not hc.comparison_operator(estimate, hc.constraint_value):
                    logger.info(
                        f"Trial {trial.number} pruned, because {estimator.metric_id} trial meets constraint: {hc.comparison_operator}({estimate}, {hc.constraint_value})"
                    )
                    raise optuna.TrialPruned()

            soft_constraints = constraint_registry.get_soft_constraints(estimator)
            for sc in soft_constraints:
                raw_value = sc.penalty_fn(
                    sc.estimate_transform(estimate), sc.constraint_value
                )

                # a reward is handled as a negative penalty
                penalty_amount = -raw_value if sc.is_reward else raw_value
                soft_score -= penalty_amount * sc.weight

                if penalty_amount:
                    logger.info(
                        f"Trial {trial.number} get a soft penalty of {penalty_amount}, because the {estimator.metric_id} is {estimate}"
                    )

        return soft_score

    return objective(trial)


def search(
    search_space_cfg: dict,
    search_algorithm: SearchAlgorithm,
    constraint_registry: ConstraintRegistry,
    hw_nas_parameters: HWNASParameters,
) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    """
    Returns: top-models, model-parameters, metrics
    """

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
            constraint_registry=constraint_registry,
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
    metric_names = [
        estimator.metric_id for estimator in constraint_registry.get_estimators()
    ]

    for frozen_trial in top_k_frozen_trials:
        top_k_models.append(search_space.create_model_sample(frozen_trial))
        top_k_params.append(frozen_trial.params)
        top_k_metrics.append(
            {
                "soft_score": eval(frozen_trial),
            }
        )
        for metric_name in metric_names:
            top_k_metrics[-1][metric_name] = frozen_trial.user_attrs[metric_name]
    return top_k_models, top_k_params, top_k_metrics
