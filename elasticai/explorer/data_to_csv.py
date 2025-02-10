import json

import pandas
import plotly.express as px

from elasticai.explorer.config import ExperimentConfig
from elasticai.explorer.explorer import Explorer
from settings import MAIN_EXPERIMENT_DIR


def build_search_space_measurements_file(latencies: list[int], explorer: Explorer) -> pandas.DataFrame:
    metrics = explorer._metric_dir / "metrics.json"
    models = explorer._model_dir / "models.json"
    with open(metrics, "r") as f:
        metric_list = json.load(f)

    with open(models, "r") as f:
        sample_list = json.load(f)

    dataframe = pandas.DataFrame.from_dict(metric_list)
    dataframe2 = pandas.DataFrame.from_dict(sample_list)

    data_merged = dataframe2.merge(dataframe, left_index=True, right_index=True)
    data_merged["latency in us"] = latencies

    csv_path = explorer._experiment_dir / "experiment_data.csv"
    data_merged.to_csv(csv_path)

    return data_merged


def plot_parallel_coordinates(df: pandas.DataFrame):
    fig = px.parallel_coordinates(
        df,
        color="default",
        color_continuous_scale=px.colors.diverging.Tealrose,
    )
    fig.show()


def read_csv(csv_path) -> pandas.DataFrame:
    return pandas.read_csv(csv_path)


if __name__ == "__main__":

    experiment_name = str(input("To plot csv data, give experiment name: "))
    csv_path = MAIN_EXPERIMENT_DIR / experiment_name / "experiment_data.csv"
    df = read_csv(csv_path)
    plot_parallel_coordinates(df)
