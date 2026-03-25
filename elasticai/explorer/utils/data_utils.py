import json
import os
from pathlib import Path
from typing import Any
import pandas as pd
import torch
from torchvision import datasets


def save_list_to_json(list: list, path_to_dir: Path, filename: str):
    os.makedirs(path_to_dir, exist_ok=True)
    with open(path_to_dir / filename, "w+") as outfile:
        json.dump(list, outfile)


def save_to_toml(toml_str: str, path_to_dir: Path, filename: str):
    os.makedirs(path_to_dir, exist_ok=True)
    with open(path_to_dir / filename, "w") as outfile:
        outfile.write(toml_str + "\n")


def load_json(path_to_json: Path | str) -> Any:
    with open(path_to_json, "r") as f:
        return json.load(f)


def read_csv(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def setup_mnist_for_cpp(root_dir_mnist: Path, root_dir_cpp_mnist: Path, transf: Any):
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
    with open(root_dir_cpp_mnist / "mnist_features.h", "w") as f:
        f.write("#ifndef MNIST_FEATURES_H\n#define MNIST_FEATURES_H\n\n")
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

        f.write("};\n\n#endif // MNIST_FEATURES_H\n")

    with open(root_dir_cpp_mnist / "mnist_labels.h", "w") as f:
        f.write("#ifndef MNIST_LABELS_H\n#define MNIST_LABELS_H\n\n")
        f.write("const int mnist_labels[256] = {\n  ")
        f.write(", ".join(str(l) for l in labels))
        f.write("\n};\n\n#endif // MNIST_LABELS_H\n")


def torch_to_tflite_sample(
    torch_sample: torch.Tensor,
) -> torch.Tensor:
    # TFlite needs an other input shape than pytorch. E.g. with N = Number of Batches, H = Height,  W = Width and C = Channels;
    # A Torch sample with NCHW Order and has to be permuted to NHWC.

    if len(torch_sample.shape) == 4:
        tflite_samples = torch_sample.permute(0, 2, 3, 1)
    elif len(torch_sample.shape) == 3:
        tflite_samples = torch_sample.permute(0, 2, 1)
    else:
        tflite_samples = torch_sample

    return tflite_samples
