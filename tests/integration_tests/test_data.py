import os
from pathlib import Path
from typing import Callable, Optional, Union
import pandas as pd
from torch import optim

from elasticai.explorer.training.data import (
    DatasetSpecification,
    MultivariateTimeseriesDataset,
)
from elasticai.explorer.training.trainer import SupervisedTrainer

from tests.integration_tests.samples.sample_MLP import SampleMLP


class TimeSeriesDatasetExample(MultivariateTimeseriesDataset):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(
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
            dataset=TimeSeriesDatasetExample(root=self.sample_dir / "test_dataset.csv"),
            deployable_dataset_path=self.sample_dir / "test_dataset.csv",
            train_val_test_ratio=[0.6, 0.2, 0.2],
        )
        model = SampleMLP(2)

        mlp_trainer = SupervisedTrainer(
            device="cpu",
            dataset_spec=dataset_spec,
            batch_size=2,
        )
        mlp_trainer.configure_optimizer(optim.Adam(model.parameters(), lr=1e-3))
        mlp_trainer.train(model, epochs=2)

        metrics, loss = mlp_trainer.validate(model)
        assert metrics["accuracy"] >= 0
        assert loss >= 0
