from scipy.stats import kendalltau
from typing import Any


def compute_kendall(list_x: list[Any], list_y: list[Any]) -> Any:
    """Computes Kendall Correlation Coefficient between list_x and list_y.

    Args:
        list_x: list of numeric values
        list_y: list of numeric values

    Returns:
        Any: the correlation coefficient
    """

    # Taking values from the above example in Lists
    rank_x = [sorted(list_x).index(x) for x in list_x]
    rank_y = [sorted(list_y).index(x) for x in list_y]

    # Calculating Kendall Rank correlation
    corr, _ = kendalltau(rank_x, rank_y)

    return corr
