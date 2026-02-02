from abc import abstractmethod
from logging import Logger
import logging
from numbers import Number
from typing import Any, Literal, Optional

import torch
from fvcore.nn.jit_handles import get_shape
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.optim.adam import Adam
from elasticai.explorer.training.trainer import Trainer


def get_values(vals: list[Any]) -> Optional[list[Any]]:
    return [v.toIValue() for v in vals]


def lstm_flop_jit(inputs: list[Any], outputs: list[Any]) -> Number:
    num_timesteps, batch_size, feature_width = get_shape(inputs[0])
    *_, proj_size = get_shape(outputs[1])
    *_, hidden_size = get_shape(outputs[2])

    *_, _, num_layers, _, _, bidirectional, batch_first = get_values(inputs)
    num_directions = 2 if bidirectional else 1
    sigmoid_flops = 1
    tanh_flops = 1
    gate_flops = 2 * (feature_width * hidden_size + hidden_size * hidden_size)
    all_gate_flops = 4 * gate_flops
    hadamard = 3 * hidden_size
    activations = 3 * sigmoid_flops * hidden_size + 2 * tanh_flops * hidden_size
    flops = (
        num_timesteps
        * num_directions
        * num_layers
        * batch_size
        * (all_gate_flops + hadamard + activations)
    )
    return flops


class Estimator:
    def __init__(self, metric_name, logger_name) -> None:
        self.metric_name: str = metric_name
        self.logger: Logger = logging.getLogger(logger_name)

    @abstractmethod
    def estimate(self, model_sample) -> tuple[float | int, list[float | int]]:
        """
        Returns:
            tuple[float|int, list[float|int]]: The final estimate and list of intermediate values.
        """
        pass


class FLOPsEstimator(Estimator):
    def __init__(self, data_sample: torch.Tensor):
        super().__init__(
            metric_name="flops_estimate", logger_name="explorer.FlopsEstimator"
        )
        self.data_sample = data_sample

    def estimate(
        self, model_sample: torch.nn.Module
    ) -> tuple[float | int, list[float | int]]:
        handlers = {"aten::sigmoid": None, "aten::lstm": lstm_flop_jit}
        flops = FlopCountAnalysis(model_sample, self.data_sample).set_op_handle(
            **handlers
        )

        return flops.total(), []


class ParamEstimator(Estimator):
    def __init__(self):
        super().__init__(
            metric_name="parameter_count_estimate",
            logger_name="explorer.ParamEstimator",
        )
        self.logger: Logger = logging.getLogger()

    def estimate(
        self, model_sample: torch.nn.Module
    ) -> tuple[float | int, list[float | int]]:
        param_count = parameter_count(model_sample)[""]
        return param_count, []


class TrainMetricsEstimator(Estimator):
    def __init__(
        self,
        trainer: Trainer,
        metric_name: str = "loss",
        n_estimation_epochs: int = 3,
    ) -> None:

        super().__init__(
            metric_name=metric_name, logger_name="explorer.TrainingEstimator"
        )
        self.trainer = trainer
        self.n_estimation_epochs = n_estimation_epochs

    def estimate(
        self, model_sample: torch.nn.Module
    ) -> tuple[float | int, list[float | int]]:
        optimizer = Adam(model_sample.parameters(), lr=1e-3)
        self.trainer.configure_optimizer(optimizer)

        estimate_values = []
        for i in range(self.n_estimation_epochs):
            self.trainer.train_epoch(model_sample, i)
            metric_avg, loss = self.trainer.validate(model_sample)

            if self.metric_name == "loss":
                estimate_value = loss
            else:
                estimate_value = metric_avg.get(self.metric_name)

            if not estimate_value:
                err = TypeError(
                    f"Trainer Type does not support {self.metric_name} estimation."
                )
                self.logger.error(
                    "%s",
                    err,
                )
                raise err

            estimate_values.append(estimate_value)

        self.logger.info(f"Estimated {self.metric_name} is: {estimate_values[-1]:.2f}")
        return estimate_values[-1], estimate_values[:-1]
