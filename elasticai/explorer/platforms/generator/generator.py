import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal
import numpy
import torch
from torch import nn
import ai_edge_torch
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch._export import capture_pre_autograd_graph

import ai_edge_torch
from ai_edge_torch.quantize.pt2e_quantizer import get_symmetric_quantization_config
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
from ai_edge_torch.quantize.quant_config import QuantConfig


class Generator(ABC):
    @abstractmethod
    def generate(
        self,
        model: nn.Module,
        path: Path,
        quantization: Literal["int8"] | Literal["full_precision"] = "full_precision",
    ) -> Any:
        pass


class PIGenerator(Generator):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.platforms.generator.generator.PIGenerator"
        )

    def generate(self, model: nn.Module, path: Path, quantization: Literal["int8"] | Literal["full_precision"] = "full_precision"):
        self.logger.info("Generate torchscript model from %s", model)
        model.eval()

        dir_path = os.path.dirname(os.path.realpath(path))

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model.to("cpu")
        ts_model = torch.jit.script(model)
        path = Path(os.path.realpath(path)).with_suffix(".pt")
        self.logger.info("Save model to %s", path)
        ts_model.save(path)

        return ts_model


class RP2040Generator(Generator):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.platforms.generator.generator.RP2040GeneratorFullPrecision"
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

    def generate(
        self,
        model: nn.Module,
        path: Path,
        quantization: Literal["int8"] | Literal["full_precision"] = "full_precision",
    ):
        self.logger.info("Generate torchscript model from %s", model)
        sample_inputs = (torch.ones(1, 784),)
        torch_output = model(*sample_inputs)
        model = model.eval()
        if quantization == "full_precision":
            edge_model = ai_edge_torch.convert(model, sample_args=sample_inputs)
        else:
            edge_model, torch_output = self._quantize(model, sample_inputs)

        edge_output = edge_model(*sample_inputs)
        self._validate(torch_output, edge_output)
        edge_model.export(str(path))


class RP2040GeneratorInt8Quantization(Generator):
    def __init__(self):
        self.logger = logging.getLogger(
            "explorer.platforms.generator.generator.RP2040GeneratorInt8Quantization"
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

        pt2e_drq_model = ai_edge_torch.convert(
            pt2e_torch_model,
            sample_input,
            quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer),
        )
        sample_input_int8 = (sample_input[0].to(torch.int8),)
        edge_output = pt2e_drq_model(*sample_input_int8)
        self.logger.debug(f"Sample output quantized: ", edge_output)
        return pt2e_drq_model

    def generate(self, model: nn.Module, path: Path):
        self.logger.info("Generate torchscript model from %s", model)
        sample_inputs = (torch.ones(1, 784),)
        quant_model = model.eval()
        quant_model = self._quantize(quant_model, sample_inputs)
        quant_model.export(str(path))
