from abc import abstractmethod
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Type, Union
import numpy as np
import pandas as pd
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torchvision.transforms import Compose

logger = logging.getLogger("explorer.data")


class BaseDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable[..., Any]] = None,
        target_transform: Optional[Callable[..., Any]] = None,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Any:
        pass


class MNISTWrapper(BaseDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable[..., Any]] = None,
        target_transform: Optional[Callable[..., Any]] = None,
        download: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(root, transform, target_transform, *args, **kwargs)
        self.dataset = MNIST(
            root=self.root,
            transform=self.transform,
            target_transform=self.target_transform,
            download=download,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[idx]


class MultivariateTimeseriesDataset(BaseDataset):
    """
    Base class for time series datasets with multiple features per time step and label.
    A feature itself should not have any channels.
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(root, transform, target_transform, *args, **kwargs)
        self.data = self._setup_data()
        self.targets = self._setup_targets()

        if len(self.data.index) != len(self.targets):
            raise ValueError("The features and labels must have the same length.")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        feature_vector = np.array(self.data.iloc[idx])
        target = np.array(self.targets.iloc[idx])
        if self.transform:
            feature_vector = self.transform(feature_vector)
        if self.target_transform:
            target = self.target_transform(target)
        return feature_vector, target

    @abstractmethod
    def _setup_data(self) -> Any:
        """Set self.data as a pandas dataset without target"""
        pass

    @abstractmethod
    def _setup_targets(self) -> Any:
        """Set self.targets as a pandas series without features"""
        pass

    @staticmethod
    def read_data(file_path: Path) -> pd.DataFrame:

        match file_path.suffix:
            case ".feather":
                return pd.read_feather(file_path)
            case ".csv":
                return pd.read_csv(file_path)
            case ".json":
                return pd.read_json(file_path)
            case _:
                raise ValueError(f"Unsupported filetype: {file_path}")


@dataclass
class DatasetSpecification:
    dataset_type: Type[BaseDataset]
    dataset_location: Path
    deployable_dataset_path: Path | None = (
        None  # This should be the path to the data that is deployed to the target device.
    )
    transform: Compose | None = None
    target_transform: Compose | None = None
    train_val_test_ratio: List[float] = field(default_factory=lambda: [0.7, 0.1, 0.2])
    shuffle: bool = False
    split_seed: int = 42
