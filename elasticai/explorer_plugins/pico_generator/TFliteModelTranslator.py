from elasticai.explorer.generator.model_translator.model_translator import (
    ModelTranslator,
)
from elasticai.explorer.hw_nas.search_space.quantization import QuantizationScheme
from elasticai.explorer_plugins.pico_generator.generator_utils import (
    torch_to_tflite_sample,
)


import ai_edge_torch
import numpy
import tensorflow as tf
import torch
from ai_edge_torch import convert, to_channel_last_io
from torch import Tensor, nn


import logging
import subprocess
from pathlib import Path


class TFliteModelTranslator(ModelTranslator):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.generator.model_translator.model_translator.TFliteModelTranslator"
        )

    def _validate(self, torch_output, edge_output):
        if numpy.allclose(
            torch_output.detach().numpy(),
            edge_output,
            atol=1e-2,
            rtol=1e-2,
        ):
            self.logger.info(
                "Inference result with Pytorch and TfLite was within tolerance."
            )
        else:
            self.logger.warning("Something wrong with Pytorch --> TfLite")

    def _quantize(
        self,
        model: nn.Module,
        sample_input: tuple[Tensor, ...],
        quantization_scheme: QuantizationScheme,
    ):

        # This only repeats the same sample, because the converter does not accept different samples.
        def representative_sample_generator():
            for _ in range(100):
                yield list(sample_input)

        if quantization_scheme.dtype == "int8":
            tfl_converter_flags = {
                "optimizations": [tf.lite.Optimize.DEFAULT],
                "representative_dataset": representative_sample_generator,
                "target_spec": {"supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]},
                "inference_input_type": tf.int8,
                "inference_output_type": tf.int8,
            }
        elif quantization_scheme.dtype == "float16":
            tfl_converter_flags = {
                "optimizations": [tf.lite.Optimize.DEFAULT],
                "representative_dataset": representative_sample_generator,
                "target_spec": {"supported_types": [tf.float16]},
                "inference_input_type": tf.float16,
                "inference_output_type": tf.float16,
            }
        else:
            err = NotImplementedError(
                f"The quantization scheme -{quantization_scheme}- is not supported by the TFliteModelTranslator."
            )
            self.logger.error(err)
            raise err
        edge_model = convert(
            model, sample_input, _ai_edge_converter_flags=tfl_converter_flags
        )

        return edge_model

    def _tflite_to_cpp_array(self, tflite_model_path: Path):
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
        quantization_scheme: QuantizationScheme | None = None,
    ):
        self.logger.info("Generate tflite model from %s", model)

        tflite_samples = torch_to_tflite_sample(sample)
        model.eval()
        torch_output = model(sample)
        tflite_shaped_model = to_channel_last_io(model, args=[0]).eval()
        if quantization_scheme:
            edge_model = self._quantize(
                tflite_shaped_model, (tflite_samples,), quantization_scheme
            )
        else:
            edge_model = ai_edge_torch.convert(
                tflite_shaped_model, sample_args=(tflite_samples,)
            )
            edge_output = edge_model(tflite_samples)
            self._validate(torch_output, edge_output)

        edge_model.export(str(output_path.with_suffix(".tflite")))
        self._tflite_to_cpp_array(output_path.with_suffix(".tflite"))
