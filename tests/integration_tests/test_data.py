import os
from pathlib import Path
import pandas as pd
from elasticai.explorer.data import FlatSequencialDataset
from torch.utils.data import DataLoader


class TestData:
    def setup_class(self):
        self.data = pd.DataFrame(
            [[1, 2, 1], [3, 4, 2], [5, 6, 3], [7, 8, 4]],
            columns=["A", "B", "labels_test"],
        )
        self.sample_dir = Path("tests/integration_tests/samples")
        os.makedirs(self.sample_dir, exist_ok=True)
        self.data.to_csv(self.sample_dir / "data.csv")

    def test_flat_sequencial_dataset(self):
        dataset = FlatSequencialDataset(
            self.sample_dir / "data.csv",
            label_names="labels_test",
        )
        assert len(dataset) == 4

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        for data, target in dataloader:
            assert len(data) == 2
            assert len(target) == 2
    

    def teardown_method(self):
        os.remove(self.sample_dir / "data.csv")
