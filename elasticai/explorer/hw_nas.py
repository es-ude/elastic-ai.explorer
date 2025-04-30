import logging
import math
from typing import Any, Type

import nni

import torch
from nni.nas import strategy
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
from nni.experiment import TrialResult


from elasticai.explorer import data
from elasticai.explorer.config import HWNASConfig
from elasticai.explorer.cost_estimator import FlopsEstimator
from elasticai.explorer.trainer import Trainer

logger = logging.getLogger("explorer.nas")


def evaluate_model(
    model: torch.nn.Module,
    device: str,
    dataset_info: data.DatasetInfo,
    trainer_class: Type[Trainer],
):
    global accuracy
    ##Parameter
    flops_weight = 3.0
    n_epochs = 2

    ##Cost-Estimation
    # flops as proxy metric for latency
    flops_estimator = FlopsEstimator()
    flops = flops_estimator.estimate_flops(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # type: ignore

    trainer = trainer_class(device, optimizer, dataset_info)

    metric = {"default": 0, "accuracy": 0, "flops log10": math.log10(flops)}
    for epoch in range(n_epochs):
        trainer.train_epoch(model, epoch)

        metric["accuracy"] = trainer.validate(model)

        metric["default"] = metric["accuracy"] - (metric["flops log10"] * flops_weight)
        nni.report_intermediate_result(metric)

    metric["accuracy"] = trainer.test(model)
    nni.report_final_result(metric)


def search(
    search_space: Any,
    hwnas_cfg: HWNASConfig,
    dataset_info: data.DatasetInfo,
    trainer_class: Type[Trainer],
) -> tuple[list[Any], list[Any], list[Any]]:
    """
    Returns: top-models, model-parameters, metrics
    """

    search_strategy = strategy.Random()
    evaluator = FunctionalEvaluator(
        evaluate_model,
        device=hwnas_cfg.host_processor,
        dataset_info=dataset_info,
        trainer_class=trainer_class,
    )
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
