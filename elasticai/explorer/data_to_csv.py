import json
from pathlib import Path
import sys

import pandas
import plotly.express as px

import settings

def build_search_space_measurements_file(latencies):
    metrics = settings.experiment_dir / "metrics/metrics.json"
    models = settings.experiment_dir / "models/models.json"
    with open(metrics, "r") as f:
        metric_list = json.load(f)

    with open(models, "r") as f:
        sample_list = json.load(f)

    dataframe = pandas.DataFrame.from_dict(metric_list)
    dataframe2 = pandas.DataFrame.from_dict(sample_list)

    data_merged = dataframe2.merge(dataframe, left_index=True, right_index=True)
    data_merged["latency in us"] = latencies

    csv_path = settings.experiment_dir / "experiment_data.csv"
    data_merged.to_csv(csv_path)

    return data_merged


def plot_parallel_coordinates(df):
    fig = px.parallel_coordinates(
        df,
        color="default",
        color_continuous_scale=px.colors.diverging.Tealrose,
    )
    fig.show()


def read_csv(filepath):
    return pandas.read_csv(filepath)


if __name__ == "__main__":
    filepath = sys.argv[0]
    df = read_csv(filepath)
    plot_parallel_coordinates(df)
