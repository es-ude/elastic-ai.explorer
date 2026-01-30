import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
from typing import Any
import ai_edge_torch
import numpy


import torch
from torch import nn
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch._export import capture_pre_autograd_graph

from ai_edge_torch import convert, to_channel_last_io
from ai_edge_torch.quantize.pt2e_quantizer import get_symmetric_quantization_config
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
from ai_edge_torch.quantize.quant_config import QuantConfig

from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
    FullPrecisionScheme,
    QuantizationScheme,
)
import elasticai.creator.nn as creator_nn
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.system_integrations.firmware_env5 import FirmwareENv5

from elasticai.creator.nn import fixed_point


class ModelCompiler(ABC):
    @abstractmethod
    def compile(
        self,
        model: nn.Module,
        output_path: Path,
        input_sample: torch.Tensor,
        quantization_scheme: QuantizationScheme,
    ) -> Any:
        pass


class TorchscriptModelCompiler(ModelCompiler):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.generator.model_compiler.model_compiler.TorchscriptModelCompiler"
        )

    def compile(
        self,
        model: nn.Module,
        output_path: Path,
        input_sample: torch.Tensor,
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


class TFliteModelCompiler(ModelCompiler):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.generator.model_compiler.model_compiler.TFliteModelCompiler"
        )

    def _validate(self, torch_output, edge_output):
        if numpy.allclose(
            torch_output.detach().numpy(),
            edge_output,
            atol=1e-2,
            rtol=1e-2,
        ):
            self.logger.info(
                "Inference result with Pytorch and TfLite was within tolerance"
            )
        else:
            self.logger.warning("Something wrong with Pytorch --> TfLite")

    def _quantize(self, model: nn.Module, sample_input: tuple[Any]):
        pt2e_quantizer = PT2EQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=False, is_dynamic=False)
        )

        pt2e_torch_model = capture_pre_autograd_graph(model, sample_input)
        pt2e_torch_model = prepare_pt2e(pt2e_torch_model, pt2e_quantizer)  # type:ignore

        # Prepare model by running one inference.
        pt2e_torch_model(*sample_input)
        pt2e_torch_model = convert_pt2e(pt2e_torch_model, fold_quantize=False)
        torch_output = pt2e_torch_model(*sample_input)

        pt2e_drq_model = convert(
            pt2e_torch_model,
            sample_input,
            quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer),
        )
        sample_input_int8 = (sample_input[0].to(torch.int8),)
        edge_output = pt2e_drq_model(*sample_input_int8)
        self.logger.debug(f"Sample output quantized: ", edge_output)
        return pt2e_drq_model, torch_output, edge_output

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

    def compile(
        self,
        model: nn.Module,
        output_path: Path,
        input_sample: torch.Tensor,
        quantization_scheme: QuantizationScheme = FullPrecisionScheme(),
    ):
        self.logger.info("Generate torchscript model from %s", model)
        input_sample_nchw = input_sample.unsqueeze(1)
        input_tuple_nchw = (input_sample_nchw,)
        input_tuple_nhwc = (input_sample_nchw.permute(0, 2, 3, 1),)
        model.eval()
        torch_output = model(*input_tuple_nchw)
        nhwc_model = to_channel_last_io(model, args=[0]).eval()
        sample_tflite_input = input_tuple_nhwc
        edge_output = None
        if isinstance(quantization_scheme, FullPrecisionScheme):
            edge_model = ai_edge_torch.convert(
                nhwc_model, sample_args=sample_tflite_input
            )
            edge_output = edge_model(*sample_tflite_input)
        elif isinstance(quantization_scheme, FixedPointInt8Scheme):
            edge_model, torch_output, edge_output = self._quantize(
                model, input_tuple_nchw
            )
            self.logger.warning(
                "Int8 quantization is supported but cannot be tested and deployed with current version of the Explorer."
            )
        else:
            err = NotImplementedError(
                f"The quantization scheme -{quantization_scheme}- is not supported by the TFliteModelCompiler."
            )
            self.logger.error(err)
            raise err

        self._validate(torch_output, edge_output)
        edge_model.export(str(output_path.with_suffix(".tflite")))
        self._model_to_cpp(output_path.with_suffix(".tflite"))


class CreatorModelCompiler(ModelCompiler):
    def __init__(self) -> None:
        self.logger = logging.getLogger(
            "explorer.generator.model_compiler.model_compiler.CreatorModelCompiler"
        )
        self.skeleton_id = [2 for i in range(16)]

    def compile(
        self,
        model: nn.Module,
        output_path: Path,
        input_sample: torch.Tensor,
        quantization_scheme: QuantizationScheme = FixedPointInt8Scheme(),
    ):
        destination = OnDiskPath(str(output_path), parent="")
        features_in = len(input_sample)
        features_out = len(model(input_sample))
        if not isinstance(model, creator_nn.Sequential):
            err = TypeError(
                f"{type(model)} is not supported by the CreatorModelCompiler, best to build models with the CreatorModelBuilder!"
            )
            self.logger.error(err)
            raise err

        my_design = model.create_design("myNetwork")

        my_design.save_to(destination.create_subpath("srcs"))

        firmware = FirmwareENv5(
            network=my_design,
            x_num_values=features_in,
            y_num_values=features_out,
            id=self.skeleton_id,
            skeleton_version="v2",
        )
        firmware.save_to(destination)

    def get_supported_layers(self) -> tuple[type] | None:
        return (fixed_point.Linear,)

    def get_supported_quantization_schemes(
        self,
    ) -> tuple[type[QuantizationScheme]] | None:
        return (FixedPointInt8Scheme,)
