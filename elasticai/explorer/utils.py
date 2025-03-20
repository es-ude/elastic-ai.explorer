import json
import os
from pathlib import Path

import pandas
import plotly.express as px
from scipy.stats import kendalltau


def compute_kendall(list_x: list[any], list_y: list[any]) -> float:
    """Computes Kandell Correlation Coefficient between list_x and list_y.

    Args:
        list_x: list of numeric values
        list_y: list of numeric values

    Returns:
        float: the correlation coeficient
    """

    # Taking values from the above example in Lists
    rank_x = [sorted(list_x).index(x) for x in list_x]
    rank_y = [sorted(list_y).index(x) for x in list_y]

    # Calculating Kendall Rank correlation
    corr, _ = kendalltau(rank_x, rank_y)

    return corr

def save_list_to_json(result: list, dir: Path, filename: str):
    os.makedirs(dir, exist_ok=True)
    with open(dir/ filename, 'w+') as outfile:
        json.dump(result, outfile)

def load_json(path_to_json: Path|str) -> any:
     with open(path_to_json, "r") as f:
            return json.load(f)


def plot_parallel_coordinates(df: pandas.DataFrame):
    fig = px.parallel_coordinates(
        df,
        color="default",
        color_continuous_scale=px.colors.diverging.Tealrose,
    )
    fig.show()