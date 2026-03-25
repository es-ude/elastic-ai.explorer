import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
from typing import Any
import ai_edge_torch
import numpy

from sympy import im
import torch
from torch import Tensor, nn

from ai_edge_torch import convert, to_channel_last_io

from elasticai.explorer.hw_nas.search_space.quantization import (
    CreatorFixedPointScheme,
    PTQFullyQuantizedInt8Scheme,
    FullPrecisionScheme,
    QuantizationScheme,
)
import tensorflow as tf

from elasticai.explorer.training.data import BaseDataset
from elasticai.explorer.utils.data_utils import torch_to_tflite_sample
from torch.utils.data import DataLoader


class ModelTranslator(ABC):
    @abstractmethod
    def translate(
        self,
        model: nn.Module,
        output_path: Path,
        sample: torch.Tensor,
        quantization_scheme: QuantizationScheme,
    ) -> Any:
        pass


class TorchscriptModelTranslator(ModelTranslator):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.generator.model_translator.model_translator.TorchscriptModelTranslator"
        )

    def translate(
        self,
        model: nn.Module,
        output_path: Path,
        sample: torch.Tensor,
        quantization_scheme: QuantizationScheme = FullPrecisionScheme(),
    ):
        if not isinstance(quantization_scheme, FullPrecisionScheme):
            err = NotImplementedError(
                f"Only Full Precision is currently not supported and not {quantization_scheme}"
            )
            self.logger.error(err)
            raise err
        self.logger.info("Generate torchscript model from %s", model)
        model.eval()

        dir_path = os.path.dirname(os.path.realpath(output_path))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model.to("cpu")
        ts_model = torch.jit.script(model)
        output_path = Path(os.path.realpath(output_path)).with_suffix(".pt")
        self.logger.info("Save model to %s", output_path)
        ts_model.save(output_path)  # type: ignore

        return ts_model


class TFliteModelTranslator(ModelTranslator):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.generator.model_translator.model_translator.TFliteModelTranslator"
        )

    def _validate(self, torch_output, edge_output, atol=1e-2, rtol=1e-2):
        if numpy.allclose(
            torch_output.detach().numpy(),
            edge_output,
            atol=atol,
            rtol=rtol,
        ):
            self.logger.info(
                "Inference result with Pytorch and TfLite was within tolerance."
            )
        else:
            self.logger.warning("Something wrong with Pytorch --> TfLite")

    def _quantize(self, model: nn.Module, sample_input: tuple[Tensor, ...]):

        # This only repeats the same sample, because the converter does not accept different samples.
        def representative_sample_generator():
            for _ in range(100):
                yield list(sample_input)

        tfl_converter_flags = {
            "optimizations": [tf.lite.Optimize.DEFAULT],
            "representative_dataset": representative_sample_generator,
            "target_spec": {"supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]},
            "inference_input_type": tf.int8,
            "inference_output_type": tf.int8,
        }
        edge_model = convert(
            model, sample_input, _ai_edge_converter_flags=tfl_converter_flags
        )

        return edge_model

    def _model_to_cpp(self, tflite_model_path: Path):
        process = subprocess.run(
            ["xxd", "-i", str(tflite_model_path)], capture_output=True
        )
        output_lines: list[str] = process.stdout.decode("utf8").splitlines(
            keepends=True
        )

        output_path = tflite_model_path.parent / tflite_model_path.stem

        with open(output_path.with_suffix(".cpp"), "w") as out_file:
            out_file.writelines("#include <model.h>\n")
            out_file.writelines(
                (
                    "const unsigned char model_tflite[] = {"
                    if line.startswith("unsigned char")
                    else line
                )
                for line in output_lines[:-1]
            )
            out_file.writelines(
                f"const unsigned int model_tflite_len = {output_lines[-1].split()[-1]}"
            )

    def translate(
        self,
        model: nn.Module,
        output_path: Path,
        sample: torch.Tensor,
        quantization_scheme: QuantizationScheme = FullPrecisionScheme(),
    ):
        self.logger.info("Generate tflite model from %s", model)

        tflite_samples = torch_to_tflite_sample(sample)
        model.eval()
        torch_output = model(sample)
        tflite_shaped_model = to_channel_last_io(model, args=[0]).eval()

        if isinstance(quantization_scheme, FullPrecisionScheme):
            edge_model = ai_edge_torch.convert(
                tflite_shaped_model, sample_args=(tflite_samples,)
            )
            edge_output = edge_model(tflite_samples)
            self._validate(torch_output, edge_output)
        elif isinstance(quantization_scheme, PTQFullyQuantizedInt8Scheme):
            edge_model = self._quantize(tflite_shaped_model, (tflite_samples,))
        else:
            err = NotImplementedError(
                f"The quantization scheme -{quantization_scheme}- is not supported by the TFliteModelTranslator."
            )
            self.logger.error(err)
            raise err

        edge_model.export(str(output_path.with_suffix(".tflite")))
        self._model_to_cpp(output_path.with_suffix(".tflite"))
