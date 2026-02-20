from dataclasses import dataclass
import logging
from typing import Any, Callable
from functools import partial

import optuna
from optuna.trial import FrozenTrial, TrialState
from optuna.study import MaxTrialsCallback
from optuna.samplers import BaseSampler

from elasticai.explorer.hw_nas.optimization_criteria import (
    OptimizationCriteria,
)
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace

logger = logging.getLogger("explorer.nas")
intermediate_metrics_template = "{metric_name}_intermediates"


@dataclass
class HWNASParameters:
    max_search_trials: int = 2
    top_n_models: int = 2
    count_only_completed_trials: bool = False

def objective_wrapper(
    trial: optuna.Trial,
    search_space_cfg: dict[str, Any],
    optimization_criteria: OptimizationCriteria,
) -> float:

    def objective(trial: optuna.Trial) -> float:

        search_space = SearchSpace(search_space_cfg)
        try:
            model = search_space.create_model_sample(trial)
        except NotImplementedError:
            raise optuna.TrialPruned()
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
                    f"Trial {trial.number} added a objective value of {objective_value:.2f}, because the {estimator.metric_name} is {final_estimate:.2f}."
                )

        logger.info(f"Trial {trial.number} has an final score of {score:.2f}")

        return score

    return objective(trial)


def search(
    search_space_cfg: dict,
    sampler: BaseSampler,
    optimization_criteria: OptimizationCriteria,
    hw_nas_parameters: HWNASParameters,
) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    """
    Returns: top-models, model-parameters, metrics
    """

    search_space = SearchSpace(search_space_cfg)
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
            optimization_criteria=optimization_criteria,
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
        estimator.metric_name for estimator in optimization_criteria.get_estimators()
    ]

    for frozen_trial in top_k_frozen_trials:
        top_k_models.append(search_space.create_model_sample(frozen_trial))
        top_k_params.append(frozen_trial.params)
        top_k_metrics.append(
            {
                "score": eval(frozen_trial),
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
