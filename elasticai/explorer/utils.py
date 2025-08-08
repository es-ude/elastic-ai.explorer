import json
import os
from pathlib import Path

import pandas
import plotly.express as px
from scipy.stats import kendalltau
from torchvision import datasets, transforms
from torchvision.transforms import transforms

from settings import ROOT_DIR


def compute_kendall(list_x: list[any], list_y: list[any]) -> float:
    """Computes Kendall Correlation Coefficient between list_x and list_y.

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


def save_list_to_json(list: list, path_to_dir: Path, filename: str):
    os.makedirs(path_to_dir, exist_ok=True)
    with open(path_to_dir / filename, "w+") as outfile:
        json.dump(list, outfile)


def load_json(path_to_json: Path | str) -> any:
    with open(path_to_json, "r") as f:
        return json.load(f)


def plot_parallel_coordinates(df: pandas.DataFrame):
    fig = px.parallel_coordinates(
        df,
        color="default",
        color_continuous_scale=px.colors.diverging.Tealrose,
    )
    fig.show()


def setup_mnist_for_cpp(root_dir_mnist: str, root_dir_cpp_mnist: str):

    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_test = datasets.MNIST(
        root=root_dir_mnist, train=False, download=True, transform=transf
    )
    images = []
    labels = []

    for i in range(256):
        img, label = mnist_test[i]
        images.append(img.squeeze().numpy())
        labels.append(label)

    os.makedirs(root_dir_cpp_mnist, exist_ok=True)
    with open(root_dir_cpp_mnist + "/mnist_images.h", "w") as f:
        f.write("#ifndef MNIST_IMAGES_H\n#define MNIST_TEST_IMAGES_H\n\n")
        f.write("const float mnist_images[256][784] = {\n")

        for img in images:
            flat = img.flatten()
            f.write("  {\n    ")
            for i in range(784):
                f.write(f"{flat[i]:.6f}f")
                if i < 783:
                    f.write(", ")
                if (i + 1) % 16 == 0 and i != 783:
                    f.write("\n    ")
            f.write("\n  },\n")

        f.write("};\n\n#endif // MNIST_IMAGES_H\n")

    with open(root_dir_cpp_mnist + "/mnist_labels.h", "w") as f:
        f.write("#ifndef MNIST_LABELS_H\n#define MNIST_LABELS_H\n\n")
        f.write("const int mnist_labels[256] = {\n  ")
        f.write(", ".join(str(l) for l in labels))
        f.write("\n};\n\n#endif // MNIST_LABELS_H\n")
