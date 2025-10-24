from abc import abstractmethod
from logging import Logger
import logging
from typing import Type
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.optim.adam import Adam
from elasticai.explorer.training.data import DatasetSpecification
from elasticai.explorer.training.trainer import Trainer

handlers = {"aten::sigmoid": None}


class Estimator:
    def __init__(self) -> None:
        self.metric_name: str = "estimate"
        self.logger: Logger = logging.getLogger("explorer.Estimator")

    @abstractmethod
    def estimate(self, model_sample) -> float | int: ...


class FLOPsEstimator(Estimator):
    def __init__(self, data_sample: torch.Tensor):
        super().__init__()
        self.metric_name = "flops_estimate"
        self.logger: Logger = logging.getLogger("explorer.FlopsEstimator")
        self.data_sample = data_sample

    def estimate(self, model_sample: torch.nn.Module) -> float:
        flops = FlopCountAnalysis(model_sample, self.data_sample).set_op_handle(
            **handlers
        )
        return flops.total()


class ParamEstimator(Estimator):
    def __init__(self):
        super().__init__()
        self.metric_name = "parameter_count_estimate"
        self.logger: Logger = logging.getLogger("explorer.ParamEstimator")

    def estimate(self, model_sample: torch.nn.Module) -> int:
        return parameter_count(model_sample)[""]


class AccuracyEstimator(Estimator):
    def __init__(
        self,
        trainer_type: Type[Trainer],
        dataset_spec: DatasetSpecification,
        n_estimation_epochs: int = 3,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.logger: Logger = logging.getLogger("explorer.AccuracyEstimator")
        self.metric_name = "accuracy_estimate"
        self.trainer_type = trainer_type
        self.n_estimation_epochs = n_estimation_epochs
        self.device = device
        self.dataset_spec = dataset_spec

    def estimate(self, model_sample: torch.nn.Module) -> float:
        optimizer = Adam(model_sample.parameters(), lr=1e-3)
        trainer = self.trainer_type(self.device, optimizer, self.dataset_spec)
        trainer.train(model_sample, self.n_estimation_epochs)
        val_accuracy, _ = trainer.validate(model_sample)
        if not val_accuracy:
            self.logger.error(
                "%s",
                TypeError("Trainer Type does not support accuracy estimation."),
            )
            raise TypeError("Trainer Type does not support accuracy estimation.")
        else:
            self.logger.info(f"Estimated Accuracy is: {val_accuracy:.2f}")
        return val_accuracy


class LossEstimator(Estimator):
    def __init__(
        self,
        trainer_type: Type[Trainer],
        dataset_spec: DatasetSpecification,
        n_estimation_epochs: int = 3,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.logger: Logger = logging.getLogger("explorer.LossEstimator")
        self.metric_name = "loss_estimate"
        self.trainer_type = trainer_type
        self.n_estimation_epochs = n_estimation_epochs
        self.device = device
        self.dataset_spec = dataset_spec

    def estimate(self, model_sample: torch.nn.Module) -> float:
        optimizer = Adam(model_sample.parameters(), lr=1e-3)
        trainer = self.trainer_type(self.device, optimizer, self.dataset_spec)
        trainer.train(model_sample, self.n_estimation_epochs)
        _, val_loss = trainer.validate(model_sample)
        self.logger.info(f"Estimated Loss is: {val_loss}")

        return val_loss
