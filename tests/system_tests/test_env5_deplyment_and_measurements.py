from pathlib import Path
from typing import Any, Callable

import torch
from elasticai.explorer.training.data import BaseDataset, DatasetSpecification
from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)

BATCH_SIZE = 64
INPUT_DIM = 6


class DatasetExample(BaseDataset):

    def __init__(
        self,
        root: str | Path,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(root, transform, target_transform, *args, **kwargs)

        self.test_data = torch.randn(BATCH_SIZE * 10, INPUT_DIM) * 5
        self.targets = torch.empty(BATCH_SIZE * 10, dtype=torch.long).random_(4)

    def __getitem__(self, idx) -> Any:
        data, target = self.test_data[idx], self.targets[idx]
        if self.transform is not None:
            data = self.transform(self.test_data[idx])

        if self.target_transform is not None:
            target = self.target_transform(self.targets[idx])
        return data, target

    def __len__(self) -> int:
        return len(self.test_data)



def create_example_dataset_spec(quantization_scheme):

    fxp_params = FxpParams(
        total_bits=quantization_scheme.total_bits,
        frac_bits=quantization_scheme.frac_bits,
        signed=quantization_scheme.signed,
    )
    fxp_conf = FxpArithmetic(fxp_params)
    return DatasetSpecification(
        dataset_type=DatasetExample,
        dataset_location=Path(""),
        deployable_dataset_path=None,
        transform=lambda x: fxp_conf.as_rational(fxp_conf.cut_as_integer(x)),
        target_transform=None,
    )


