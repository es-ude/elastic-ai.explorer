import logging
import math
from typing import Any, Callable
from functools import partial

import optuna
from optuna.trial import FrozenTrial, TrialState
from optuna.study import MaxTrialsCallback
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace

from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.hw_nas.cost_estimator import FlopsEstimator
from elasticai.explorer.trainer import MLPTrainer

logger = logging.getLogger("explorer.nas")


def objective_wrapper(
    trial: optuna.Trial,
    search_space_cfg: dict[str, Any],
    device: str,
    n_estimation_epochs: int,
    flops_weight: float,
) -> float:

    def objective(trial: optuna.Trial) -> float:

        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_loader = DataLoader(
            MNIST("data/mnist", download=True, transform=transf),
            batch_size=64,
            shuffle=True,
        )
        test_loader = DataLoader(
            MNIST("data/mnist", download=True, train=False, transform=transf),
            batch_size=64,
        )

        search_space = SearchSpace(search_space_cfg)
        model = search_space.create_model_sample(trial)
        trainer = MLPTrainer(device, optimizer=Adam(model.parameters(), lr=1e-3))

        flops_estimator = FlopsEstimator()
        sample, _ = next(iter(train_loader))
        flops = flops_estimator.estimate_flops(model, sample)
        metric = {"default": 0, "accuracy": 0, "flops log10": math.log10(flops)}

        for epoch in range(n_estimation_epochs):
            trainer.train_epoch(model, train_loader, epoch)

            metric["accuracy"] = trainer.test(model, test_loader)

            metric["default"] = metric["accuracy"] - (
                metric["flops log10"] * flops_weight
            )
            trial.report(metric["default"], epoch)
        trial.set_user_attr("accuracy", metric["accuracy"])
        trial.set_user_attr("flops_log10", metric["flops log10"])
        return metric["default"]

    return objective(trial)


def search(
    search_space_cfg: dict, hwnas_cfg: HWNASConfig
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
                device=hwnas_cfg.host_processor,
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
                "accuracy": frozen_trial.user_attrs["accuracy"],
                "flops log10": frozen_trial.user_attrs["flops_log10"],
            }
        )

    return top_k_models, top_k_params, top_k_metrics
