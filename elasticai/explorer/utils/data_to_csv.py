from pathlib import Path
import pandas as pd

from elasticai.explorer.utils import data_utils
from elasticai.explorer.utils.data_utils import read_csv
from elasticai.explorer.utils.visualize import plot_parallel_coordinates


def build_search_space_measurements_file(
    metric_to_measurements: dict,
    metrics_path: Path,
    model_parameter_path: Path,
    csv_path: Path,
) -> pd.DataFrame:
    metric_list = data_utils.load_json(metrics_path)
    sample_list = data_utils.load_json(model_parameter_path)

    dataframe = pd.DataFrame.from_dict(metric_list)
    dataframe2 = pd.DataFrame.from_dict(sample_list)

    data_merged = dataframe2.merge(dataframe, left_index=True, right_index=True)

    for metric, measurements in metric_to_measurements.items():

        data_merged[metric] = measurements

    data_merged.to_csv(csv_path)

    return data_merged


if __name__ == "__main__":
    experiment_path = Path(input("To plot csv data, give path to experiment: "))
    csv_path = experiment_path / "experiment_data.csv"
    df = read_csv(csv_path)
    plot_parallel_coordinates(df)
