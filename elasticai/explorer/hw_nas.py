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
    flops_estimator = FlopsEstimator(model_space=model)
    flops = flops_estimator.estimate_flops_single_module()
    mlp_trainer = MLPTrainer(device="cpu", optimizer= torch.optim.Adam(model.parameters(), lr=1e-3))


    #create test and dataloader from MNIST #TODO create Dataset Class
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainloader = DataLoader(
        MNIST("data/mnist", download=True, transform=transf),
        batch_size=64,
        shuffle=True,
    )
    testloader = DataLoader(
        MNIST("data/mnist", download=True, train=False, transform=transf), batch_size=64
    )
        
    metric = {"default": 0, "accuracy": 0, "flops log10": math.log10(flops)}

    for epoch in range(n_epochs):
        mlp_trainer.train_epoch(model=model, trainloader=trainloader, epoch=epoch)
        metric["accuracy"] = mlp_trainer.test(model=model, testloader=testloader)
        metric["default"] = metric["accuracy"] - (metric["flops log10"] * flops_weight)
        nni.report_intermediate_result(metric)

    nni.report_final_result(metric)


def search(search_space, max_search_trials=6, top_k=4):
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
