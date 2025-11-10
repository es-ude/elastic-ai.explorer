from abc import abstractmethod
from logging import Logger
import logging
from numbers import Number
from typing import Any, List, Optional, Type

import torch
from fvcore.nn.jit_handles import get_shape
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.optim.adam import Adam
from elasticai.explorer.training.data import DatasetSpecification
from elasticai.explorer.training.trainer import Trainer, accuracy_fn


def get_values(vals: List[Any]) -> Optional[List[Any]]:
    return [v.toIValue() for v in vals]


def lstm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
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
    def estimate(self, model_sample) -> float | int: ...


class FLOPsEstimator(Estimator):
    def __init__(self, data_sample: torch.Tensor):
        super().__init__(
            metric_name="flops_estimate", logger_name="explorer.FlopsEstimator"
        )
        self.data_sample = data_sample

    def estimate(self, model_sample: torch.nn.Module) -> float:
        handlers = {"aten::sigmoid": None, "aten::lstm": lstm_flop_jit}
        flops = FlopCountAnalysis(model_sample, self.data_sample).set_op_handle(
            **handlers
        )
        return flops.total()


class ParamEstimator(Estimator):
    def __init__(self):
        super().__init__(
            metric_name="parameter_count_estimate",
            logger_name="explorer.ParamEstimator",
        )
        self.logger: Logger = logging.getLogger()

    def estimate(self, model_sample: torch.nn.Module) -> int:
        return parameter_count(model_sample)[""]


class TrainingEstimator(Estimator):
    def __init__(
        self,
        trainer: Trainer,
        metric_name: str,
        logger_name: str,
        n_estimation_epochs: int = 3,
    ) -> None:
        super().__init__(metric_name=metric_name, logger_name=logger_name)
        self.trainer = trainer
        self.n_estimation_epochs = n_estimation_epochs


class AccuracyEstimator(TrainingEstimator):
    def __init__(
        self,
        trainer: Trainer,
        n_estimation_epochs: int = 3,
    ) -> None:
        super().__init__(
            trainer=trainer,
            metric_name="accuracy_estimate",
            logger_name="explorer.AccuracyEstimator",
            n_estimation_epochs=n_estimation_epochs,
        )

    def estimate(self, model_sample: torch.nn.Module) -> float:
        optimizer = Adam(model_sample.parameters(), lr=1e-3)
        self.trainer.configure_optimizer(optimizer)
        self.trainer.train(model_sample, self.n_estimation_epochs)
        metric_avg, _ = self.trainer.validate(model_sample)
        validation_accuracy = metric_avg.get("accuracy")
        if not validation_accuracy:
            self.logger.error(
                "%s",
                TypeError("Trainer Type does not support accuracy estimation."),
            )
            raise TypeError("Trainer Type does not support accuracy estimation.")
        else:
            self.logger.info(f"Estimated Accuracy is: {validation_accuracy:.2f}")

        return validation_accuracy


class LossEstimator(TrainingEstimator):
    def __init__(
        self,
        trainer: Trainer,
        n_estimation_epochs: int = 3,
    ) -> None:
        super().__init__(
            trainer=trainer,
            metric_name="loss_estimate",
            logger_name="explorer.LossEstimator",
            n_estimation_epochs=n_estimation_epochs,
        )

    def estimate(self, model_sample: torch.nn.Module) -> float:
        optimizer = Adam(model_sample.parameters(), lr=1e-3)
        self.trainer.configure_optimizer(optimizer)
        self.trainer.train(model_sample, self.n_estimation_epochs)
        _, val_loss = self.trainer.validate(model_sample)
        self.logger.info(f"Estimated Loss is: {val_loss}")

        return val_loss
