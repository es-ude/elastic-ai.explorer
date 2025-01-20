import json

import pandas
import plotly.express as px

from settings import ROOT_DIR


def build_search_space_measurements_file(latencies):
    metrics = str(ROOT_DIR) + "/metrics/metrics.json"
    models = str(ROOT_DIR) + "/models/models.json"
    with open(metrics, "r") as f:
        metric_list = json.load(f)

    with open(models, "r") as f:
        sample_list = json.load(f)

    dataframe = pandas.DataFrame.from_dict(metric_list)
    dataframe2 = pandas.DataFrame.from_dict(sample_list)

    data_merged = dataframe2.merge(dataframe, left_index=True, right_index=True)
    data_merged["latency in us"] = latencies

    csv_path = str(ROOT_DIR) + "/experiment_data.csv"
    data_merged.to_csv(csv_path)

    return data_merged


def plot_parallel_coordinates(df):
    fig = px.parallel_coordinates(
        df,
        color="default",
        color_continuous_scale=px.colors.diverging.Tealrose,
    )
    fig.show()


def read_csv():
    return pandas.read_csv(str(ROOT_DIR) + "/experiment_data_comp_sp.csv")


if __name__ == "__main__":
    df = read_csv()
    plot_parallel_coordinates(df)
