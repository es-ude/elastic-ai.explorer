import os
from pathlib import Path
from typing import Any

import torch
from torchvision import datasets

from pathlib import Path
import os
import numpy as np
from typing import Any, Sequence


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


def _write_header_guard(f, name: str):
    guard = name.upper()
    f.write(f"#ifndef {guard}\n#define {guard}\n\n")


def _close_header_guard(f, name: str):
    guard = name.upper()
    f.write(f"\n#endif // {guard}\n")


def export_features_to_header(
    features: Sequence[np.ndarray],
    output_path: Path,
    var_name: str = "features",
    dtype: str = "float",
    flatten: bool = True,
):
    """
    Export feature tensors to a C++ header file.

    Args:
        features: list/array of images (H,W) or (C,H,W) or already flat
        output_path: .h file path
        var_name: C++ variable name
        dtype: 'float' or 'int8'
        flatten: whether to flatten input tensors
    """

    features = [np.array(f) for f in features]

    if flatten:
        features = [f.flatten() for f in features]

    num_samples = len(features)
    feature_len = features[0].size

    with open(output_path, "w") as f:
        _write_header_guard(f, var_name + "_H")

        f.write(f"const {dtype} {var_name}[{num_samples}][{feature_len}] = {{\n")

        for sample in features:
            f.write("  {\n    ")

            for i, val in enumerate(sample):
                if dtype == "float":
                    f.write(f"{float(val):.6f}f")
                else:
                    f.write(f"{int(val)}")

                if i < feature_len - 1:
                    f.write(", ")

                if (i + 1) % 16 == 0 and i != feature_len - 1:
                    f.write("\n    ")

            f.write("\n  },\n")

        f.write("};\n")
        _close_header_guard(f, var_name + "_H")


def export_labels_to_header(
    labels: Sequence[int],
    output_path: Path,
    var_name: str = "labels",
    dtype: str = "int",
):

    num_samples = len(labels)

    with open(output_path, "w") as f:
        _write_header_guard(f, var_name + "_H")

        f.write(f"const {dtype} {var_name}[{num_samples}] = {{\n  ")
        f.write(", ".join(str(int(l)) for l in labels))
        f.write("\n};\n")

        _close_header_guard(f, var_name + "_H")


def prepare_image_dataset_for_cpp(
    dataset,
    output_dir: Path,
    num_samples: int = 256,
    transform: Any = None,
    dtype: str = "float",
    flatten: bool = True,
):
    """
    Generic image-dataset exporter for C++ (TFLite Micro compatible).

    Args:
        dataset: torchvision-style dataset
        output_dir: directory for headers
        num_samples: number of samples to export
        transform: optional transform override
        dtype: 'float' or 'int8'
        flatten: flatten images or keep original shape
    """

    images = []
    labels = []

    for i in range(num_samples):
        img, label = dataset[i]

        if transform is not None:
            img = transform(img)

        # Convert to numpy
        if hasattr(img, "numpy"):
            img = img.numpy()
        else:
            img = np.array(img)

        images.append(img)
        labels.append(label)

    os.makedirs(output_dir, exist_ok=True)

    export_features_to_header(
        images,
        output_dir / "features.h",
        var_name="features",
        dtype=dtype,
        flatten=flatten,
    )

    export_labels_to_header(
        labels,
        output_dir / "labels.h",
        var_name="labels",
    )
