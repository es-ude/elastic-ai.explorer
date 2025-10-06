import logging
import math
from typing import Any, Callable, Type
from functools import partial

import optuna
from optuna.trial import FrozenTrial, TrialState

from torch.optim.adam import Adam

from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from elasticai.explorer.training import data
from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.hw_nas.cost_estimator import FlopsEstimator
from elasticai.explorer.training.trainer import Trainer, TrainerFactory

logger = logging.getLogger("explorer.nas")


def objective_wrapper(
    trial: optuna.Trial,
    search_space_cfg: dict[str, Any],
    trainer_cls: Trainer,
    n_estimation_epochs: int,
    flops_weight: float,
) -> float:

    def objective(trial: optuna.Trial) -> float:

        search_space = SearchSpace(search_space_cfg)
        model = search_space.create_model_sample(trial)
        trainer = trainer_cls.create_instance()
        optimizer = Adam(model.parameters(), lr=1e-3)  # type: ignore
        trainer.configure_optimizer(optimizer)

        # trainer = trainer_class(device, optimizer, dataset_spec)

        flops_estimator = FlopsEstimator()
        sample, _ = next(iter(trainer.test_loader))
        flops = flops_estimator.estimate_flops(model, sample)
        metric = {
            "default": 0,
            "val_loss": 0,
            "val_accuracy": -1,
            "flops log10": math.log10(flops),
        }
        for epoch in range(n_estimation_epochs):
            trainer.train_epoch(model, epoch)
            val_metrics, val_loss = trainer.validate(model)
            metric["val_loss"] = val_loss
            if "accuracy" in val_metrics:
                metric["val_accuracy"] = val_metrics["accuracy"] * 100
                metric["default"] = metric["val_accuracy"] - (
                    metric["flops log10"] * flops_weight
                )
            else:
                metric["val_accuracy"] = -1
                metric["default"] = -metric["val_loss"] - (
                    metric["flops log10"] * flops_weight
                )
            trial.report(metric["default"], epoch)
        trial.set_user_attr("val_accuracy", metric["val_accuracy"])
        trial.set_user_attr("flops_log10", metric["flops log10"])
        trial.set_user_attr("val_loss", metric["val_loss"])
        return metric["default"]

    return objective(trial)


def search(
    search_space_cfg: dict,
    hwnas_cfg: HWNASConfig,
    trainer: Trainer,
) -> tuple[list[Any], list[dict[str, Any]], list[Any]]:
    """
    Returns: top-models, model-parameters, metrics
    """

    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(),
        direction="maximize",
    )
    study.optimize(
        partial(
            objective_wrapper,
            search_space_cfg=search_space_cfg,
            trainer_cls=trainer,
            n_estimation_epochs=hwnas_cfg.n_estimation_epochs,
            flops_weight=hwnas_cfg.flops_weight,
        ),
        n_trials=hwnas_cfg.max_search_trials,
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
    search_space = SearchSpace(search_space_cfg)

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
