import os
from pathlib import Path
from typing import Callable, Optional, Union
import pandas as pd
from websockets import Data
from elasticai.explorer.training.data import (
    DatasetInfo,
    FlatSequencialDataset,
)
import torch

from elasticai.explorer.training.download import DownloadableSciebo
from elasticai.explorer.training.trainer import MLPTrainer
from tests.integration_tests.samples.sample_MLP import SampleMLP
from iesude.data.archives import PlainFile


class SequentialTestDataset(FlatSequencialDataset, DownloadableSciebo):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        download_path = root
        super().__init__(
            download=download,
            download_path=download_path,
            file_path_on_sciebo="test_dataset.csv",
            file_type=PlainFile,
            root=str(root),
            train=train,
            transform=transform,
            target_transform=target_transform,
        )

    def _setup_data(self):
        self.data: pd.DataFrame = self.read_data(Path(self.root)).drop(
            "labels_test", axis="columns"
        )

    def _setup_targets(self):
        self.targets: pd.Series = self.read_data(Path(self.root))["labels_test"]


class TestData:
    def setup_class(self):
        self.sample_dir = Path("tests/integration_tests/samples")
        os.makedirs(self.sample_dir, exist_ok=True)
        self.dataset_info = DatasetInfo(
            SequentialTestDataset, self.sample_dir / "test_dataset.csv", None
        )

    def test_flat_sequencial_dataset(self):
        dataset = SequentialTestDataset(
            root=self.sample_dir / "test_dataset.csv", download=True
        )
        assert len(dataset) == 21

    def test_flat_sequencial_dataset_mlp_trainer(self):

        model = SampleMLP(2)

        mlp_trainer = MLPTrainer(
            device="cpu",
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),  # type : ignore
            dataset_info=self.dataset_info,
            batch_size=2,
        )
        mlp_trainer.train(model, epochs=2)
        mlp_trainer.validate(model)

    def teardown_method(self):
        os.remove(self.sample_dir / "test_dataset.csv")
