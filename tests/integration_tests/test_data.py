import os
from pathlib import Path
from typing import Callable, Optional, Union
from numpy import dtype
import pandas as pd
from elasticai.explorer.data import DatasetInfo, FlatSequencialDataset
import torch
from torch.utils.data import DataLoader

from elasticai.explorer.trainer import MLPTrainer
from elasticai.explorer.utils import get_file_from_sciebo
from iesude.data.archives import PlainFile

from tests.integration_tests.samples.sample_MLP import SampleMLP


class SequentialTestDataset(FlatSequencialDataset):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, train, transform, target_transform, download)

    def _download_data(self):
        get_file_from_sciebo(
            save_dir=str(self.root),
            file_path_in_sciebo="test_dataset.csv",
            file_type=PlainFile,  # type: ignore
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

    def test_flat_sequencial_dataset(self):
        dataset = SequentialTestDataset(
            self.sample_dir / "test_dataset.csv", download=True
        )
        assert len(dataset) == 21

    def test_flat_sequencial_dataset_mlp_trainer(self):

        model = SampleMLP(2)

        mlp_trainer = MLPTrainer(
            device="cpu",
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),  # type: ignore
            dataset_info=DatasetInfo(
                SequentialTestDataset, self.sample_dir / "test_dataset.csv", None
            ),
            batch_size=2,
        )
        mlp_trainer.train(model, epochs=2)
        mlp_trainer.validate(model)

    def teardown_method(self):
        os.remove(self.sample_dir / "test_dataset.csv")
