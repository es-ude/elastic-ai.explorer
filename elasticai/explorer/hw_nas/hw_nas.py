import logging
import math
from typing import Any, Type

import nni

import torch

from nni.nas import strategy
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
from nni.nas.nn.pytorch import ModelSpace
from nni.experiment import TrialResult
from nni.nas.nn.pytorch import ModelSpace

from elasticai.explorer.training import data
from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.hw_nas.cost_estimator import FlopsEstimator
from elasticai.explorer.training.trainer import Trainer

logger = logging.getLogger("explorer.nas")


def evaluate_model(
    model: ModelSpace,
    device: str,
    dataset_spec: data.DatasetSpecification,
    trainer_class: type[Trainer],
):
    global accuracy
    flops_weight = 3.0
    n_epochs = 2

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # type: ignore

    trainer = trainer_class(device, optimizer, dataset_spec)
    flops_estimator = FlopsEstimator()
    sample, _ = next(iter(trainer.test_loader))
    flops = flops_estimator.estimate_flops(model, sample)

    metric = {
        "default": 0,
        "val_loss": 0,
        "val_accuracy": 0,
        "flops log10": math.log10(flops),
    }
    for epoch in range(n_epochs):
        trainer.train_epoch(model, epoch)
        val_accuracy, val_loss = trainer.validate(model)
        metric["val_loss"] = val_loss
        if val_accuracy:
            metric["val_accuracy"] = val_accuracy
            metric["default"] = metric["val_accuracy"] - (
                metric["flops log10"] * flops_weight
            )
        else:
            metric["val_accuracy"] = -1
            metric["default"] = -metric["val_loss"] - (
                metric["flops log10"] * flops_weight
            )
        nni.report_intermediate_result(metric)

    nni.report_final_result(metric)


def search(
    search_space: Any,
    hwnas_cfg: HWNASConfig,
    dataset_spec: data.DatasetSpecification,
    trainer_class: Type[Trainer],
) -> tuple[list[Any], list[Any], list[Any]]:

    search_strategy = strategy.Random()
    evaluator = FunctionalEvaluator(
        evaluate_model,
        device=hwnas_cfg.host_processor,
        dataset_spec=dataset_spec,
        trainer_class=trainer_class,
    )
    experiment = NasExperiment(search_space, evaluator, search_strategy)
    experiment.config.max_trial_number = hwnas_cfg.max_search_trials
    experiment.run(port=8081)
    top_models: list[ModelSpace] = experiment.export_top_models(
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
    for model in top_models:
        print("Simplify")
        print(model.simplify())
        print("Parameters")
        print(model.parameters())
        print(metrics)
        print(parameters)
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
