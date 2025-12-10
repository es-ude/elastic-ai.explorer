from abc import abstractmethod
import logging
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count

handlers = {"aten::sigmoid": None}


class Estimator:
    def __init__(self) -> None:
        self.metric_name: str = "estimate"
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self.logger = logging.getLogger(logger_name)

    @abstractmethod
    def estimate(self, model_sample) -> float | int: ...


class FLOPsEstimator(Estimator):
    def __init__(self, data_sample: torch.Tensor):
        super().__init__()
        self.metric_name = "flops_estimate"
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

    def estimate(self, model_sample: torch.nn.Module) -> int:
        return parameter_count(model_sample)[""]


if __name__ == "__main__":
    test = FLOPsEstimator(torch.tensor([1]))
