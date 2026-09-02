import subprocess
from pathlib import Path
from typing import Literal

import litert_torch
import numpy as np
import torch
from litert_torch.quantize import PT2EQuantizer
from litert_torch.quantize.pt2e_quantizer import get_symmetric_quantization_config
from litert_torch.quantize.quant_config import QuantConfig
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

from elasticai.explorer.platforms.generator.generator import Generator


class LitertGenerator(Generator):
    #     def create_sample_input(self, input_sample):
    #         input_sample_nchw = input_sample.unsqueeze(1)
    #         input_tuple_nchw = (input_sample_nchw,)
    #         input_tuple_nhwc = (input_sample_nchw.permute(0, 2, 3, 1),)
    #
    #         torch_output = model(*input_tuple_nchw)
    #         nhwc_model = to_channel_last_io(model, args=[0]).eval()
    #         sample_tflite_input = input_tuple_nhwc

    def _model_to_cpp(self, tflite_model_path: Path):
        cpp_path = tflite_model_path.with_suffix(".cpp")

        with cpp_path.open("w") as out_file:
            out_file.write('#include "model.h"\n')

            subprocess.run(
                ["xxd", "-i", str(tflite_model_path)],
                stdout=out_file,
                check=True,
            )

    def _quantize_torch_model(self, model: nn.Module, sample_input):
        pt2e_quantizer = PT2EQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        )
        pt2e_torch_model = torch.export.export(model, sample_input).module()

        pt2e_torch_model = prepare_pt2e(pt2e_torch_model, pt2e_quantizer)
        # Run the prepared model with sample input data to ensure that internal observers are populated with correct values
        pt2e_torch_model(*sample_input)

        # Convert the prepared model to a quantized model
        pt2e_torch_model = convert_pt2e(pt2e_torch_model, fold_quantize=False)
        return pt2e_torch_model, pt2e_quantizer

    def _generate_for_quantized_model(self, model: nn.Module, sample_input):
        quantized_torch_model, quantizer = self._quantize_torch_model(
            model, sample_input
        )

        # Convert to a litert_torch model
        pt2e_drq_model = litert_torch.convert(
            quantized_torch_model,
            sample_input,
            quant_config=QuantConfig(pt2e_quantizer=quantizer),
        )

    def generate(
        self,
        model: nn.Module,
        path: Path,
        input_sample: torch.Tensor,
        quantization: Literal["full_precision"] = "full_precision",
    ) -> any:
        path_with_suffix = path.with_suffix(".tflite")
        edge_model = litert_torch.convert(model.eval(), (input_sample,))
        edge_model.export(str(path_with_suffix))
        self._model_to_cpp(path_with_suffix)
        return edge_model
