import os
from pathlib import Path
from typing import Callable, Optional, Union
import pandas as pd
import pytest
from torch import optim

from elasticai.explorer.training.data import (
    DatasetSpecification,
    MultivariateTimeseriesDataset,
)

from elasticai.explorer.training.download import DownloadableSciebo
from elasticai.explorer.training.trainer import SupervisedTrainer

from tests.integration_tests.samples.sample_MLP import SampleMLP
from iesude.data.archives import PlainFile
import pytest


@pytest.fixture
def test_dataset_path(tmp_path):
    csv_content = """A,B,labels_test
7,8,1
7,8,1
7,8,1
7,8,1
7,8,1
7,8,1
7,8,1
7,8,1
7,8,1
7,8,1
7,8,1
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
9,8,4
"""

    csv_path = tmp_path / "test_dataset.csv"
    csv_path.write_text(csv_content)

    return csv_path


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
    # def setup_class(self):
    #     self.sample_dir = Path("tests/integration_tests/samples")
    #     os.makedirs(self.sample_dir, exist_ok=True)

    def test_dataset(self, test_dataset_path):
        dataset = TimeSeriesDatasetExample(root=test_dataset_path)
        assert len(dataset) == 27

    def test_dataset_with_mlp_trainer(self, test_dataset_path):
        dataset_spec = DatasetSpecification(
            dataset=TimeSeriesDatasetExample(root=test_dataset_path),
            deployable_dataset_path=test_dataset_path,
            train_val_test_ratio=[0.6, 0.2, 0.2],
        )
        model = SampleMLP(2)
        print(model)
        mlp_trainer = SupervisedTrainer(
            device="cpu",
            dataset_spec=dataset_spec,
            batch_size=5,
        )
        mlp_trainer.configure_optimizer(optim.Adam(model.parameters(), lr=1e-3)),
        mlp_trainer.train(model, epochs=2)

        metrics, loss = mlp_trainer.validate(model)
        assert metrics["accuracy"] >= 0
        assert loss >= 0
