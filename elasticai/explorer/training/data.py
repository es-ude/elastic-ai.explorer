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
from sklearn.model_selection import train_test_split

logger = logging.getLogger("explorer.data")


class BaseDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    @abstractmethod
    def _len_(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Any:
        pass


class FlatSequencialDataset(BaseDataset):
    """
    Base class for sequencial datasets with only 1-Dimensional features and labels.
    """

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(root, train, transform, target_transform, *args, **kwargs)
        self._setup_data()
        self._setup_targets()

        if train:
            self.data, _ = train_test_split(self.data, test_size=0.2, random_state=42)
            self.targets, _ = train_test_split(
                self.targets, test_size=0.2, random_state=42
            )
        else:
            _, self.data = train_test_split(self.data, test_size=0.2, random_state=42)
            _, self.targets = train_test_split(
                self.targets, test_size=0.2, random_state=42
            )

        if len(self.data.index) != len(self.targets):
            raise ValueError("The features and labels must have the same length.")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        feature_vector = np.array(self.data.iloc[idx, 1:])
        target = np.array(self.targets.iloc[idx])
        if self.transform:
            feature_vector = self.transform(feature_vector)
        if self.target_transform:
            target = self.target_transform(target)
        return feature_vector, target

    @abstractmethod
    def _setup_data(self):
        """Set self.data as a pandas dataset without target"""
        pass

    @abstractmethod
    def _setup_targets(self):
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
class DatasetInfo:
    dataset_type: Type[MNIST] | Type[BaseDataset]
    dataset_location: Path
    transform: Compose | None = None
    validation_split_ratio: List[float] = field(default_factory=lambda: [0.8, 0.2])
