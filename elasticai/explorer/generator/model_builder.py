from elasticai.explorer.generator.reflection import Reflective
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme


from abc import ABC, abstractmethod
from typing import Any


class ModelBuilder(Reflective, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def build_from_trial(
        self, trial, search_space: dict
    ) -> tuple[Any, QuantizationScheme | None]:
        pass
