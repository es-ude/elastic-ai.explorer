from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Type
from venv import logger
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose

logger = logging.getLogger("explorer.data")

@dataclass
class DatasetInfo:
    dataset_type: Type[Dataset]
    dataset_location: Path
    transform: Compose 


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


class FlatSequencialDataset(Dataset):
    """Represents sequencial datasets with only 1-Dimensional features and labels"""

    def __init__(
        self,
        dataset_file: Path,
        label_file: Path,
        transform=None,
        target_transform=None,
    ):
        self.features: pd.DataFrame = read_data(dataset_file)
        self.labels: pd.DataFrame = read_data(label_file)
        if len(self.features) != len(self.labels):
            raise ValueError("The features and labels must have the same length.")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vector = self.features.iloc[idx, :]
        label = self.features.iloc[idx, 1]
        if self.transform:
            feature_vector = self.transform(feature_vector)
        if self.target_transform:
            label = self.target_transform(label)
        return feature_vector, label
