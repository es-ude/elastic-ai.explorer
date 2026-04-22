from elasticai.explorer.hw_nas.search_space.build_model import DefaultModelBuilder
from typing import Any


class PicoModelBuilder(DefaultModelBuilder):
    def get_supported_quantization(self) -> dict[str, Any]:
        return {
            "dtype": {"int8"},
        }
