import logging
from elasticai.explorer.hw_nas.search_space.layer_builder import (
    parse_search_param,
)
from elasticai.explorer.hw_nas.search_space.quantization_builder import QUANT_REGISTRY
from elasticai.explorer.hw_nas.search_space.registry import (
    ADAPTER_REGISTRY,
    LAYER_REGISTRY,
)


class SearchSpace:
    def __init__(self, search_space_cfg: dict):
        self.search_space_cfg = search_space_cfg
        self.next_input_shape = search_space_cfg["input"]
        self.output_shape = search_space_cfg["output"]
        self.blocks = search_space_cfg["blocks"]
        self.layers = []
        self.logger = logging.getLogger("explorer.hw_nas.search_space")

    def is_last_block(self, block_id):
        return self.blocks[-1]["block"] == block_id

    def create_block(self, trial, block: dict, prev_operation=None):
        block_id = block["block"]
        num_layers = parse_search_param(
            trial, f"num_layers_b{block_id}", block, "depth", default_value=None
        )
        operation = parse_search_param(
            trial, f"operation_b{block_id}", block, "op_candidates", default_value=None
        )
        quantization = parse_search_param(
            trial,
            f"quant_b{block_id}",
            block,
            "quant_candidates",
            default_value="full_precision",
        )

        #    if prev_operation is not None:
        adapter_cls = ADAPTER_REGISTRY.get((prev_operation, operation))
        print(adapter_cls)
        if adapter_cls is not None:
            self.logger.info(f"Inserting adapter: {prev_operation} -> {operation}")
            adapter = adapter_cls()
            self.layers.append(adapter)
            self.next_input_shape = adapter_cls.infer_output_shape(
                self.next_input_shape
            )
        builder_cls = LAYER_REGISTRY[operation]
        quantization_builder_cls = QUANT_REGISTRY[quantization]
        quantization_scheme = quantization_builder_cls(
            trial,
            block,
            block.get(quantization, {}),
            block_id,
        ).build()

        builder = builder_cls(
            trial,
            block,
            block.get(operation, {}),
            block_id,
            self.next_input_shape,
            self.output_shape,
            quantization_scheme,
        )
        layers, out_shape = builder.build(num_layers, self.is_last_block(block_id))

        self.layers.extend(layers)
        if self.is_last_block(block_id):
            last_layer_adapter = ADAPTER_REGISTRY.get((operation, None))
            if last_layer_adapter is not None:
                last_layer_adapter = last_layer_adapter()
                self.layers.append(last_layer_adapter)
                self.next_input_shape = last_layer_adapter.infer_output_shape(
                    self.next_input_shape
                )
        self.next_input_shape = out_shape

        return operation

    def create_model_layers(self, trial) -> list:
        self.next_input_shape = self.search_space_cfg["input"]
        self.output_shape = self.search_space_cfg["output"]
        self.layers = []

        prev_operation = None
        for block in self.blocks:

            prev_operation = self.create_block(trial, block, prev_operation)

        return self.layers
