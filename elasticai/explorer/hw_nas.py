import logging
import math
from typing import Any

import nni
import torch
from nni.nas.strategy import Random
from nni.nas.strategy.middleware import Filter, Chain
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
from nni.nas.nn.pytorch import ModelSpace
from nni.experiment import TrialResult
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.cost_estimator import FlopsEstimator
from elasticai.explorer.trainer import MLPTrainer

logger = logging.getLogger("explorer.nas")


def evaluate_model(model: ModelSpace, device: str):
    global accuracy
    ##Parameter
    flops_weight = 3.0
    n_epochs = 2

    ##Cost-Estimation
    # flops as proxy metric for latency
    flops_estimator = FlopsEstimator()
    flops = flops_estimator.estimate_flops(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # type: ignore
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
    trainer = MLPTrainer(device, optimizer)

    metric = {"default": 0, "accuracy": 0, "flops log10": math.log10(flops)}
    for epoch in range(n_epochs):
        trainer.train_epoch(model, train_loader, epoch)

        metric["accuracy"] = trainer.test(model, test_loader)

        metric["default"] = metric["accuracy"] - (metric["flops log10"] * flops_weight)
        nni.report_intermediate_result(metric)

    nni.report_final_result(metric)


def search(
    search_space: Any, hwnas_cfg: HWNASConfig
) -> tuple[list[Any], list[Any], list[Any]]:
    """
    Returns: top-models, model-parameters, metrics
    """
    
    filters: list[Filter] = []

    for key, value in hwnas_cfg.hw_constraints.items():
        if key == "max_flops":
            filters.append(Filter(lambda sample: flops_estimator.estimate_flops(sample) < value))
        elif key == "max_params":
            filters.append(Filter(lambda sample: flops_estimator.compute_num_params(sample) < value))
        else:
            logger.warning("Unknown hardware constraint: %s", key)
    
    if filters.empty():
        search_strategy = Random()
    else:
        search_strategy = Chain(Random(), *filters)

    evaluator = FunctionalEvaluator(evaluate_model, device=hwnas_cfg.host_processor)
    experiment = NasExperiment(search_space, evaluator, search_strategy)
    experiment.config.max_trial_number = hwnas_cfg.max_search_trials
    experiment.run(port=8081)
    top_models = experiment.export_top_models(
        top_k=hwnas_cfg.top_n_models, formatter="instance"
    )
    top_parameters = experiment.export_top_models(
        top_k=hwnas_cfg.top_n_models, formatter="dict"
    )
    test_results = experiment.export_data()
    experiment.stop()

    metrics, parameters = _map_trial_params_to_found_models(
        test_results, top_parameters
    )

    return top_models, parameters, metrics


def _map_trial_params_to_found_models(
    test_results: list[TrialResult], top_parameters: list[Any]
):
    parameters = list(range(len(top_parameters)))
    metrics: list[Any] = list(range(len(top_parameters)))
    for trial in test_results:
        for i, top_parameter in enumerate(top_parameters):
            if trial.parameter["sample"] == top_parameter:
                parameters[i] = trial.parameter["sample"]

                metrics[i] = trial.value
    return metrics, parameters
