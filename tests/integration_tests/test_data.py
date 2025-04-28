import os
from pathlib import Path
import pandas as pd

from elasticai.explorer.data import FlatSequencialDataset


class TestData:
    def setup_class(self):
        self.features = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6], [7, 8]], columns=["A", "B"]
        )
        self.labels = pd.DataFrame([1, 2, 3, 4], columns=["C"])
        self.sample_dir = Path("tests/integration_tests/samples")
        os.makedirs(self.sample_dir, exist_ok=True)
        self.features.to_csv(self.sample_dir / "features.csv")
        self.labels.to_csv(self.sample_dir / "labels.csv")
        print(self.labels)

    def test_flat_sequencial_dataset(self):
        dataset = FlatSequencialDataset(
            self.sample_dir / "features.csv", self.sample_dir / "labels.csv"
        )
        assert len(dataset) == 4

    def teardown_method(self):
        os.remove(self.sample_dir / "features.csv")
        os.remove(self.sample_dir / "labels.csv")
        
