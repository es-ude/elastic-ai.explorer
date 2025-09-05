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


class TimeSeriesDatasetExample(MultivariateTimeseriesDataset, DownloadableSciebo):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        download_path = root
        super().__init__(
            download_path=download_path,
            file_path_in_sciebo_datasets="test_dataset.csv",
            file_type=PlainFile,
            root=str(root),
            transform=transform,
            target_transform=target_transform,
        )

    def _setup_data(self):
        data: pd.DataFrame = self.read_data(Path(self.root)).drop(
            "labels_test", axis="columns"
        )
        return data

    def _setup_targets(self):
        targets: pd.Series = self.read_data(Path(self.root))["labels_test"]
        return targets


class TestData:
    def setup_class(self):
        self.sample_dir = Path("tests/integration_tests/samples")
        os.makedirs(self.sample_dir, exist_ok=True)

    def test_dataset(self):
        dataset = TimeSeriesDatasetExample(root=self.sample_dir / "test_dataset.csv")
        assert len(dataset) == 27

    def test_dataset_with_mlp_trainer(self):
        dataset_spec = DatasetSpecification(
            dataset_type=TimeSeriesDatasetExample,
            dataset_location=self.sample_dir / "test_dataset.csv",
            deployable_dataset_path=self.sample_dir / "test_dataset.csv",
            transform=None,
            test_train_val_ratio=[0.6, 0.2, 0.2],
        )
        model = SampleMLP(2)

        mlp_trainer = MLPTrainer(
            device="cpu",
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),  # type: ignore
            dataset_spec=dataset_spec,
            batch_size=2,
        )
        mlp_trainer.train(model, epochs=2)
        accuracy, loss = mlp_trainer.validate(model)
        assert accuracy is not None
        assert accuracy >= 0
        assert loss >= 0

    def teardown_method(self):
        try:
            os.remove(self.sample_dir / "test_dataset.csv")
        except:
            pass
