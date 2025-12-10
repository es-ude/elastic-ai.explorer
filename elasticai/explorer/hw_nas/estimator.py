from abc import abstractmethod
from logging import Logger
import logging
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count

handlers = {"aten::sigmoid": None}


class Estimator:
    def __init__(self) -> None:
        self.metric_name: str = "estimate"
        self.logger: Logger = logging.getLogger("explorer.hw_nas.estimator.Estimator")

    @abstractmethod
    def estimate(self, model_sample) -> float | int: ...


class FLOPsEstimator(Estimator):
    def __init__(self, data_sample: torch.Tensor):
        super().__init__()
        self.metric_name = "flops_estimate"
        self.logger: Logger = logging.getLogger(
            "explorer.hw_nas.estimator.FlopsEstimator"
        )
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
        self.logger: Logger = logging.getLogger(
            "explorer.hw_nas.estimator.ParamEstimator"
        )

    def estimate(self, model_sample: torch.nn.Module) -> int:
        return parameter_count(model_sample)[""]
