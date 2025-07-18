import os
from pathlib import Path
from typing import Callable, Optional, Union
import pandas as pd
from elasticai.explorer.training.data import (
    DatasetSpecification,
    MultivariateTimeseriesDataset,
)
import torch

from elasticai.explorer.training.download import DownloadableSciebo
from elasticai.explorer.training.trainer import MLPTrainer
from tests.integration_tests.samples.sample_MLP import SampleMLP
from iesude.data.archives import PlainFile


class TestTimeSeriesDataset(MultivariateTimeseriesDataset, DownloadableSciebo):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        download_path = root
        super().__init__(
            download_path=download_path,
            file_path_in_sciebo_datasets="test_dataset.csv",
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
        self.dataset_spec = DatasetSpecification(
            TestTimeSeriesDataset, self.sample_dir / "test_dataset.csv", None
        )

    def test_dataset(self):
        dataset = TestTimeSeriesDataset(root=self.sample_dir / "test_dataset.csv")
        assert len(dataset) == 21

    def test_dataset_with_mlp_trainer(self):

        model = SampleMLP(2)

        mlp_trainer = MLPTrainer(
            device="cpu",
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),  # type: ignore
            dataset_spec=self.dataset_spec,
            batch_size=2,
        )
        mlp_trainer.train(model, epochs=2)
        assert mlp_trainer.validate(model)[0] >= 0

    def teardown_method(self):
        os.remove(self.sample_dir / "test_dataset.csv")
