import logging
import math

import nni
import torch
from nni.nas import strategy
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.cost_estimator import FlopsEstimator
from elasticai.explorer.trainer import MLPTrainer

logger = logging.getLogger("explorer.nas")


def evaluate_model(model: torch.nn.Module, device):
    global accuracy
    ##Parameter
    flops_weight = 3.
    n_epochs = 2

    ##Cost-Estimation
    # flops as proxy metric for latency
    flops_estimator = FlopsEstimator()
    flops = flops_estimator.estimate_flops(model)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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


def search(search_space: any, hwnas_cfg: HWNASConfig) -> tuple[list[any], list[any], list[any]]:
    """
    Returns: top-models, model-parameters, metrics
    """
    search_strategy = strategy.Random()
    evaluator = FunctionalEvaluator(evaluate_model, device=hwnas_cfg.host_processor)
    experiment = NasExperiment(search_space, evaluator, search_strategy)
    experiment.config.max_trial_number = hwnas_cfg.max_search_trials
    experiment.run(port=8081)
    top_models = experiment.export_top_models(top_k=hwnas_cfg.top_n_models, formatter="instance")
    top_parameters = experiment.export_top_models(top_k=hwnas_cfg.top_n_models, formatter="dict")
    test_results = experiment.export_data()
    experiment.stop()

    metrics, parameters = _map_trial_params_to_found_models(test_results, top_parameters)

    return top_models, parameters, metrics


def _map_trial_params_to_found_models(test_results, top_parameters):
    parameters = list(range(len(top_parameters)))
    metrics = list(range(len(top_parameters)))
    for trial in test_results:
        for i, top_parameter in enumerate(top_parameters):
            if trial.parameter["sample"] == top_parameter:
                parameters[i] = trial.parameter["sample"]

                metrics[i] = trial.value
    return metrics, parameters
