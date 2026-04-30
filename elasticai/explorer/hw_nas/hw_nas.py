import logging
import traceback
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import FrozenTrial, TrialState

from elasticai.explorer.hw_nas.optimization_criteria import (
    OptimizationCriteria,
)
from elasticai.explorer.hw_nas.search_space.build_model import (
    ShapeValueError,
    construct_model,
)
from elasticai.explorer.hw_nas.search_space.sample_blocks import Sampler

logger = logging.getLogger("explorer.nas")
intermediate_metrics_template = "{metric_name}_intermediates"


@dataclass
class HWNASParameters:
    max_search_trials: int = 2
    top_n_models: int = 2
    count_only_completed_trials: bool = False


class SearchStrategy(Enum):
    RANDOM_SEARCH = "random"
    EVOLUTIONARY_SEARCH = "evolution"


def _evaluate_constraints(trial, model, optimization_criteria: OptimizationCriteria):
    score = 0.0
    for estimator in optimization_criteria:
        final_estimate, estimates = estimator.estimate(model)
        trial.set_user_attr(estimator.metric_name, final_estimate)
        trial.set_user_attr(
            intermediate_metrics_template.format(metric_name=estimator.metric_name),
            estimates,
        )
        hard_constraints = optimization_criteria.get_hard_constraints(estimator)
        for hc in hard_constraints:
            if not hc.comparator(final_estimate, hc.constraint_value):
                logger.info(
                    f"Trial {trial.number} pruned, because {estimator.metric_name} trial does not meet constraint: {hc.comparator}({final_estimate:.2f}, {hc.constraint_value})."
                )
                raise optuna.TrialPruned()

        soft_constraints = optimization_criteria.get_soft_constraints(estimator)
        for sc in soft_constraints:
            if not sc.comparator(final_estimate, sc.constraint_value):
                penalty_value = sc.penalty_weight * sc.penalty_fn(
                    sc.penalty_estimate_transform(final_estimate),
                    sc.constraint_value,
                )
                score -= penalty_value
                logger.info(
                    f"Trial {trial.number} gets a soft penalty of {penalty_value:.2f}, because {estimator.metric_name} trial does not meet constraint: {sc.comparator}({final_estimate:.2f}, {sc.constraint_value})."
                )

        objectives = optimization_criteria.get_objectives(estimator)
        for o in objectives:
            if o.transform:
                objective_value = o.weight * o.transform(final_estimate)
            else:
                objective_value = o.weight * final_estimate

            score += objective_value
            logger.info(
                f"Trial {trial.number} added an objective value of {objective_value:.2f}, because the {estimator.metric_name} is {final_estimate:.2f}."
            )
    return score


def sample_and_create_model(trial, search_space: dict):
    search_space_sampler = Sampler(trial)
    try:
        sample = search_space_sampler.construct_sample(search_space)
        model = construct_model(sample, search_space["input"], search_space["output"])
        return model

    except (ShapeValueError, NotImplementedError) as e:
        print(traceback.format_exc())
        logger.warning(
            f"Failed to construct model due to exception: {e}. Pruning trial."
        )
        raise optuna.TrialPruned()


def objective_wrapper(
    trial: optuna.Trial,
    search_space_cfg: dict[str, Any],
    optimization_criteria: OptimizationCriteria,
) -> float:
    def objective(trial: optuna.Trial) -> float:
        model = sample_and_create_model(trial, search_space_cfg)
        score = _evaluate_constraints(trial, model, optimization_criteria)
        logger.info(f"Trial {trial.number} has a final score of {score:.2f}")

        return score

    return objective(trial)


def create_sampler(
    search_strategy: SearchStrategy,
) -> optuna.samplers.BaseSampler:
    match search_strategy:
        case SearchStrategy.RANDOM_SEARCH:
            sampler = optuna.samplers.RandomSampler()
        case SearchStrategy.EVOLUTIONARY_SEARCH:
            sampler = optuna.samplers.NSGAIISampler(
                population_size=20,
                mutation_prob=0.1,
            )
        case _:
            sampler = optuna.samplers.RandomSampler()
    return sampler


def create_trial_callbacks(
    hw_nas_parameters: HWNASParameters,
) -> tuple[int | None, list]:
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

    return n_trials, callbacks


def collect_top_k_results(
    study: optuna.Study,
    hw_nas_parameters: HWNASParameters,
    optimization_criteria: OptimizationCriteria,
    search_space_cfg: dict,
) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    test_results = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    trial_score: Callable[[FrozenTrial], float] = lambda trial: (
        trial.value if trial.value is not None else float("-inf")
    )
    test_results.sort(key=trial_score, reverse=True)

    top_k_frozen_trials = test_results[: hw_nas_parameters.top_n_models]

    if len(top_k_frozen_trials) == 0:
        logger.warning("No models found in the search space.")
        return [], [], []

    top_k_models: list[Any] = []
    top_k_params: list[dict[str, Any]] = []
    top_k_metrics: list[dict] = []
    metric_names = [
        estimator.metric_name for estimator in optimization_criteria.get_estimators()
    ]

    for frozen_trial in top_k_frozen_trials:
        top_k_models.append(sample_and_create_model(frozen_trial, search_space_cfg))
        top_k_params.append(frozen_trial.params)
        top_k_metrics.append(
            {
                "score": trial_score(frozen_trial),
            }
        )
        for metric_name in metric_names:
            intermediates_key = intermediate_metrics_template.format(
                metric_name=metric_name
            )
            top_k_metrics[-1][metric_name] = frozen_trial.user_attrs[metric_name]
            top_k_metrics[-1][intermediates_key] = frozen_trial.user_attrs[
                intermediates_key
            ]
    return top_k_models, top_k_params, top_k_metrics


def search(
    search_space_cfg: dict,
    search_strategy: SearchStrategy,
    optimization_criteria: OptimizationCriteria,
    hw_nas_parameters: HWNASParameters,
) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    """
    Returns: top-models, model-parameters, metrics
    """

    sampler = create_sampler(search_strategy=search_strategy)

    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
    )

    n_trials, callbacks = create_trial_callbacks(hw_nas_parameters=hw_nas_parameters)

    study.optimize(
        partial(
            objective_wrapper,
            search_space_cfg=search_space_cfg,
            optimization_criteria=optimization_criteria,
        ),
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    top_k_models, top_k_params, top_k_metrics = collect_top_k_results(
        study=study,
        hw_nas_parameters=hw_nas_parameters,
        optimization_criteria=optimization_criteria,
        search_space_cfg=search_space_cfg,
    )
    return top_k_models, top_k_params, top_k_metrics
