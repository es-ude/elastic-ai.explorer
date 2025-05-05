from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Type
from venv import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose

logger = logging.getLogger("explorer.data")


class FlatSequencialDataset(Dataset):
    """
    Represents sequencial datasets with only 1-Dimensional features and labels.
    label_names (Optional): the column name of the labels. Default = 'lables'
    """

    def __init__(
        self, dataset_file: Path, transform=None, target_transform=None, **kwargs
    ):
        label_names = kwargs.get("label_names", "labels")
        self.data: pd.DataFrame = read_data(dataset_file).drop(
            label_names, axis="columns"
        )
        self.targets: pd.Series = read_data(dataset_file)[label_names]
        if len(self.data) != len(self.targets):
            raise ValueError("The features and labels must have the same length.")

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_vector = np.array(self.data.iloc[idx, :])
        label = np.array(self.targets.iloc[idx])
        if self.transform:
            feature_vector = self.transform(feature_vector)
        if self.target_transform:
            label = self.target_transform(label)
        return feature_vector, label


@dataclass
class DatasetInfo:
    dataset_type: Type[MNIST] | Type[FlatSequencialDataset]
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
