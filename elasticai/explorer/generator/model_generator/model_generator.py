from enum import Enum
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional
import numpy

from elasticai.creator.torch2ir.torch2ir import get_default_converter
from elasticai.creator.ir2vhdl.ir2vhdl import Ir2Vhdl

import torch
from torch import nn
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch._export import capture_pre_autograd_graph

import ai_edge_torch
from ai_edge_torch.quantize.pt2e_quantizer import get_symmetric_quantization_config
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
from ai_edge_torch.quantize.quant_config import QuantConfig


class QuantizationSchemes(str, Enum):
    FULL_PRECISION_FLOAT32 = "full_precision_float32"
    INT8_UNIFORM = "int8_uniform"


class ModelGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        model: nn.Module,
        path: Path,
        input_sample: torch.Tensor,
        quantization_scheme: QuantizationSchemes = QuantizationSchemes.FULL_PRECISION_FLOAT32,
    ) -> Any:
        pass

    def get_supported_layers(self) -> Optional[set[type]]:
        """Override if necessary. "None" means no constraints."""
        return None

    def get_supported_quantization_schemes(self) -> Optional[set[QuantizationSchemes]]:
        """Override if necessary. "None" means no constraints."""
        return None

    def _validate_model(
        self, model: nn.Module, quantization_scheme: QuantizationSchemes
    ):
        """Override if necessary"""
        supported_layers = self.get_supported_layers()
        supported_quantization_schemes = self.get_supported_quantization_schemes()
        if supported_layers is not None:
            for layer in model.modules():
                if layer is model:
                    continue
                if type(layer) not in supported_layers:
                    raise NotImplementedError(
                        f"Layer {type(layer).__name__} wird von {self.__class__.__name__} nicht unterstützt"
                    )

        if supported_quantization_schemes is not None:
            if quantization_scheme not in supported_quantization_schemes:
                raise NotImplementedError(
                    f"Layer {quantization_scheme} wird von {self.__class__.__name__} nicht unterstützt"
                )


class TorchscriptModelGenerator(ModelGenerator):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.generator.generator.generator.PIGenerator"
        )

    def generate(
        self,
        model: nn.Module,
        path: Path,
        input_sample: torch.Tensor,
        quantization_scheme: QuantizationSchemes = QuantizationSchemes.FULL_PRECISION_FLOAT32,
    ):
        if quantization_scheme == QuantizationSchemes.INT8_UNIFORM:
            raise NotImplementedError(
                "int8-Uniform-Quantization is currently not supported."
            )
        self.logger.info("Generate torchscript model from %s", model)
        model.eval()

        dir_path = os.path.dirname(os.path.realpath(path))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model.to("cpu")
        ts_model = torch.jit.script(model)
        path = Path(os.path.realpath(path)).with_suffix(".pt")
        self.logger.info("Save model to %s", path)
        ts_model.save(path)  # type: ignore

        return ts_model


class TFliteModelGenerator(ModelGenerator):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.generator.generator.generator.RP2040GeneratorFullPrecision"
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

        pt2e_drq_model = ai_edge_torch.convert(
            pt2e_torch_model,
            sample_input,
            quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer),
        )
        sample_input_int8 = (sample_input[0].to(torch.int8),)
        edge_output = pt2e_drq_model(*sample_input_int8)
        self.logger.debug(f"Sample output quantized: ", edge_output)
        return pt2e_drq_model, torch_output

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

    def generate(
        self,
        model: nn.Module,
        path: Path,
        input_sample: torch.Tensor,
        quantization_scheme: QuantizationSchemes = QuantizationSchemes.FULL_PRECISION_FLOAT32,
    ):
        self.logger.info("Generate torchscript model from %s", model)

        input_sample_nchw = input_sample.unsqueeze(1)
        input_tuple_nchw = (input_sample_nchw,)
        input_tuple_nhwc = (input_sample_nchw.permute(0, 2, 3, 1),)

        torch_output = model(*input_tuple_nchw)
        nhwc_model = ai_edge_torch.to_channel_last_io(model, args=[0]).eval()
        sample_tflite_input = input_tuple_nhwc
        if quantization_scheme == QuantizationSchemes.FULL_PRECISION_FLOAT32:
            edge_model = ai_edge_torch.convert(
                nhwc_model, sample_args=sample_tflite_input
            )
        else:
            edge_model, torch_output = self._quantize(model, input_tuple_nchw)
            self.logger.warning(
                "Int8 quantization is supported but cannot be tested and deployed with current version of the Explorer."
            )

        edge_output = edge_model(*sample_tflite_input)
        self._validate(torch_output, edge_output)
        edge_model.export(str(path.with_suffix(".tflite")))
        self._model_to_cpp(path.with_suffix(".tflite"))


class CreatorModelGenerator(ModelGenerator):
    def __init__(self) -> None:
        self.logger = logging.getLogger(
            "explorer.generator.model_compiler.model_compiler.CreatorModelCompiler"
        )

    def generate(
        self,
        model: nn.Module | Dict,
        path: Path,
        input_sample: torch.Tensor,
        quantization_scheme: QuantizationSchemes = QuantizationSchemes.FULL_PRECISION_FLOAT32,
    ):

        
        # self._validate_model(model, quantization_scheme)
        # model = model
        # default_converter = get_default_converter()
        # ir = default_converter.convert(model)
        # for impl in ir:
        #     print("implementation: ", impl)
        # ir2Vhdl = Ir2Vhdl()
        # vhdl_code = ir2Vhdl(ir)  # type:ignore
        # for code in vhdl_code:
        #     print("VHDL code: ", code)

        pass

    def get_supported_layers(self) -> Optional[set[type]]:
        return {
            nn.Linear,
            nn.Conv1d,
            nn.ReLU,
            nn.MaxPool1d,
            nn.BatchNorm1d,
            nn.Flatten,
            nn.Sigmoid,
        }

    def _build_creator_model_from_parametrisation(self, params: Dict):

