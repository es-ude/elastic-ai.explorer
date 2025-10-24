from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence
from torch import nn
import math

activation_mapping = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}


class ModelBuilder(ABC):
    """
    Abstrakte Schnittstelle: baut ein nn.Module aus einer festen Parametrisierung (params dict).
    params: Dictionary mit genau den Keys, wie sie auch in Optuna FrozenTrial.params auftauchen.
    """

    @abstractmethod
    def build_from_params(
        self, params: Dict[str, Any], search_space_cfg: Dict[str, Any]
    ) -> nn.Module:
        pass


class PytorchModelBuilder(ModelBuilder):
    """
    Default-Implementierung: übersetzt die SearchSpace-Parametrisierung in ein nn.Sequential
    analog zum bisherigen Verhalten von SearchSpace.create_model_sample(trial).
    """

    def __init__(self) -> None:
        super().__init__()

    def _get_param(self, params: Dict[str, Any], name: str, fallback: Any):
        # Wenn Param existiert, gebe ihn zurück, sonst fallback (Konstanten aus search-space)
        return params.get(name, fallback)

    def _create_linear_block(
        self,
        layers: list,
        params: Dict[str, Any],
        block: Dict[str, Any],
        block_id: str,
        num_layers: int,
        input_shape: Any,
        output_shape: int,
    ) -> Any:
        # input_shape kann int oder list (flatten vorher)
        if isinstance(input_shape, (list, tuple)):
            layers.append(nn.Flatten())
            current_shape = int(math.prod(input_shape))
        else:
            current_shape = int(input_shape)

        for i in range(num_layers):
            layer_width = int(
                self._get_param(params, f"layer_width_b{block_id}_l{i}", block["linear"]["width"][0])
            )
            activation = self._get_param(
                params, f"activation_func_b{block_id}_l{i}", block.get("activation", ["relu"])[0]
            )
            # letzte Layer der gesamten Arch. mappt auf output_shape
            is_last_layer_overall = False
            # Der Caller sollte prüfen ob letzter Block und letzter Layer (hier nicht möglich zu wissen)
            # Wir legen Verhalten fest: wenn layer_width == output_shape treat as last; (keine 1:1 Entsprechung)
            if i == (num_layers - 1):
                # erzeugen linear zu layer_width (Standard). Das Gesamt-Final-Layer wird außerhalb ggf. angepasst.
                layers.append(nn.Linear(current_shape, layer_width))
            else:
                layers.append(nn.Linear(current_shape, layer_width))
            current_shape = layer_width
            layers.append(activation_mapping[activation])

        return current_shape

    def _create_conv_block(
        self,
        layers: list,
        params: Dict[str, Any],
        block: Dict[str, Any],
        block_id: str,
        num_layers: int,
        input_shape: Sequence[int],
    ) -> Sequence[int]:
        current_shape = list(input_shape)
        for i in range(num_layers):
            out_channels = int(
                self._get_param(params, f"out_channels_b{block_id}_l{i}", block["conv2D"]["out_channels"][0])
            )
            kernel_size = int(
                self._get_param(params, f"kernel_size_b{block_id}_l{i}", block["conv2D"]["kernel_size"][0])
            )
            stride = int(
                self._get_param(params, f"stride_b{block_id}_l{i}", block["conv2D"]["stride"][0])
            )
            activation = self._get_param(
                params, f"activation_func_b{block_id}_l{i}", block.get("activation", ["relu"])[0]
            )

            layers.append(nn.Conv2d(current_shape[0], out_channels, kernel_size, stride))
            layers.append(activation_mapping[activation])
            # einfache Output-Shape-Berechnung (H,W)
            c, h, w = current_shape
            new_h = (h - kernel_size) // stride + 1
            new_w = (w - kernel_size) // stride + 1
            current_shape = [out_channels, new_h, new_w]

        return current_shape

    def build_from_params(self, params: Dict[str, Any], search_space_cfg: Dict[str, Any]):
        input_shape = search_space_cfg["input"]
        output_shape = search_space_cfg["output"]
        blocks = search_space_cfg["blocks"]

        layers: list = []
        current_input = input_shape

        for block in blocks:
            block_id = str(block["block"])
            # num_layers kann als Liste (choices) oder dict mit range etc. -> params enthält finalen Wert oder use first choice
            num_layers_key = f"num_layers_b{block_id}"
            # fallback: falls depth eine Liste im YAML ist, nehme erstes Element
            depth = block.get("depth", 1)
            fallback_num_layers = depth[0] if isinstance(depth, list) else depth
            num_layers = int(params.get(num_layers_key, fallback_num_layers))

            op_key = f"operation_b{block_id}"
            fallback_op = block.get("op_candidates", ["linear"])[0]
            operation = params.get(op_key, fallback_op)

            if operation == "linear":
                # Wenn input ist bildlich und erste op linear -> flatten handled in _create_linear_block
                current_input = self._create_linear_block(
                    layers, params, block, block_id, num_layers, current_input, output_shape
                )
            elif operation == "conv2d":
                # current_input muss [C,H,W]
                if isinstance(current_input, (list, tuple)):
                    current_input = list(current_input)
                else:
                    raise ValueError("conv2d requires channel-shaped input")
                current_input = self._create_conv_block(
                    layers, params, block, block_id, num_layers, current_input
                )
            else:
                raise NotImplementedError(f"Operation {operation} not supported by PytorchModelBuilder")

        # Falls letzte Schicht nicht auf output_shape abgebildet wurde, ergänze final linear
        # Heuristik: falls aktueller shape ein int -> connect zu output_shape
        if isinstance(current_input, int):
            if current_input != output_shape:
                layers.append(nn.Linear(current_input, output_shape))
        else:
            # wenn conv output -> flatten dann linear
            if isinstance(current_input, (list, tuple)):
                layers.append(nn.Flatten())
                prod = int(math.prod(current_input))
                layers.append(nn.Linear(prod, output_shape))

        return nn.Sequential(*layers)