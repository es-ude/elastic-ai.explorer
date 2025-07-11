import logging
import math
import os
from typing import Any, Callable
from functools import partial

import optuna
from optuna.trial import FrozenTrial, TrialState
from optuna.study import MaxTrialsCallback

from elasticai.explorer.hw_nas.search_space.construct_sp import SearchSpace

from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.hw_nas.cost_estimator import FlopsEstimator
from elasticai.explorer.trainer import MLPTrainer

logger = logging.getLogger("explorer.nas")


def objective_wrapper(trial: optuna.Trial, search_space_cfg: dict[str, Any], device: str) -> float:

    def objective(trial: optuna.Trial) -> float:
        global accuracy
        flops_weight = 3.0  # TODO: make configurable
        n_epochs = 2    # TODO: make configurable

        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_loader = DataLoader(
            MNIST("data/mnist", download=True, transform=transf),
            batch_size=64,
            shuffle=True,
        )
        test_loader = DataLoader(
            MNIST("data/mnist", download=True, train=False, transform=transf), batch_size=64
        )

        search_space = SearchSpace(search_space_cfg)
        model = search_space.create_model_sample(trial)
        trainer = MLPTrainer(device, optimizer=Adam(model.parameters(), lr=1e-3))

        flops_estimator = FlopsEstimator()
        sample, _ = next(iter(train_loader))
        flops = flops_estimator.estimate_flops(model, sample)
        metric = {"default": 0, "accuracy": 0, "flops log10": math.log10(flops)}

        for epoch in range(n_epochs):
            trainer.train_epoch(model, train_loader, epoch)

            metric["accuracy"] = trainer.test(model, test_loader)

            metric["default"] = metric["accuracy"] - (metric["flops log10"] * flops_weight)
            trial.report(metric["default"], epoch)  # TODO: report accuracy, too

        return metric["default"]    # TODO: report accuracy, too
    
    return objective(trial)


def search(
    search_space: Any, hwnas_cfg: HWNASConfig
) -> tuple[list[Any], dict[str, Any], list[Any]]:
    """
    Returns: top-models, model-parameters, metrics
    """

    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(),
        direction="maximize",
    )
    study.optimize(
        partial(objective_wrapper, search_space_cfg=search_space, device=hwnas_cfg.host_processor),
        callbacks=[MaxTrialsCallback(hwnas_cfg.max_search_trials, states=(TrialState.COMPLETE,TrialState.RUNNING, TrialState.WAITING))],
        n_jobs=(os.cpu_count() // 4),  # TODO: Use user defined portion of the available CPU cores
        show_progress_bar=True,
    )

    # best_model = study.best_trial
    # best_parameters = study.best_params
    # best_metrics = study.best_value

    test_results = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    eval: Callable[[FrozenTrial], float] = lambda trial: trial.value if trial.value is not None else float("-inf")
    test_results.sort(key=eval, reverse=True)

    top_k_models = test_results[:hwnas_cfg.top_n_models]
    if len(top_k_models) == 0:
        logger.warning("No models found in the search space.")
        return [], {}, []
    
    top_k_model_numbers: list[int] = []
    top_k_params: dict[str, Any] = {}
    top_k_metrics: list[float] = []

    for model in top_k_models:
        top_k_model_numbers.append(model.number)
        top_k_params.update(model.params)
        top_k_metrics.append(eval(model))

    return top_k_models, top_k_params, top_k_metrics

