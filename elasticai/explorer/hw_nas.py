import json
import logging
import math

import nni
import torch
from nni.nas import strategy
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from elasticai.explorer.cost_estimator import FlopsEstimator
from elasticai.explorer.trainer import MLPTrainer

logger = logging.getLogger("explorer.nas")


def evaluate_model(model: torch.nn.Module):
    global accuracy
    ##Parameter
    flops_weight = 3.
    n_epochs = 2

    ##Cost-Estimation
    # flops as proxy metric for latency
    flops_estimator = FlopsEstimator()
    flops = flops_estimator.estimate_flops(model)

    # set device to cpu to prevent memory error
    device = "cpu"
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
    trainer = MLPTrainer(device,optimizer)

    metric = {"default": 0, "accuracy": 0, "flops log10": math.log10(flops)}
    for epoch in range(n_epochs):
        trainer.train_epoch(model,train_loader, epoch)

        metric["accuracy"] = trainer.test(model, test_loader)

        metric["default"] = metric["accuracy"] - (metric["flops log10"] * flops_weight)
        nni.report_intermediate_result(metric)

    nni.report_final_result(metric)


def search(search_space: any, max_search_trials: int = 6, top_k: int = 4) -> list[any]:
    search_strategy = strategy.Random()
    
    evaluator = FunctionalEvaluator(evaluate_model)
    exp = NasExperiment(search_space, evaluator, search_strategy)
    exp.config.max_trial_number = max_search_trials
    exp.run(port=8081)
    top_models = exp.export_top_models(top_k=top_k, formatter="instance")
    top_parameters = exp.export_top_models(top_k=top_k, formatter="dict")
    test_results = exp.export_data()

    # sorting the metrics, parameters in the top_k order
    parameters = list(range(len(top_parameters)))
    metrics = list(range(len(top_parameters)))
    for trial in test_results:
        for i, top_parameter in enumerate(top_parameters):
            if trial.parameter["sample"] == top_parameter:
                parameters[i] = trial.parameter["sample"]

                metrics[i] = trial.value

    with open('models/models.json', 'w') as outfile:
        json.dump(parameters, outfile)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f)

    return top_models
