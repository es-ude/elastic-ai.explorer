from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import torch
from torch import nn


from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme


class ModelTranslator(ABC):

    @abstractmethod
    def translate(
        self,
        model: nn.Module,
        output_path: Path,
        sample: torch.Tensor,
        quantization_scheme: QuantizationScheme | None = None,
    ) -> Any:
        pass
