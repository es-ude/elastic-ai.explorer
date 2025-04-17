from pathlib import Path
import pandas

from elasticai.explorer import utils
from elasticai.explorer.utils import plot_parallel_coordinates
from settings import MAIN_EXPERIMENT_DIR


def build_search_space_measurements_file(
    latencies: list[int], metrics_path: Path, model_parameter_path: Path, csv_path: Path
) -> pandas.DataFrame:
    metric_list = utils.load_json(metrics_path)
    sample_list = utils.load_json(model_parameter_path)

    dataframe = pandas.DataFrame.from_dict(metric_list)
    dataframe2 = pandas.DataFrame.from_dict(sample_list)

    data_merged = dataframe2.merge(dataframe, left_index=True, right_index=True)
    data_merged["latency in us"] = latencies

    data_merged.to_csv(csv_path)

    return data_merged


def read_csv(csv_path) -> pandas.DataFrame:
    return pandas.read_csv(csv_path)


if __name__ == "__main__":
    experiment_name = str(input("To plot csv data, give experiment name: "))
    csv_path = MAIN_EXPERIMENT_DIR / experiment_name / "experiment_data.csv"
    df = read_csv(csv_path)
    plot_parallel_coordinates(df)
