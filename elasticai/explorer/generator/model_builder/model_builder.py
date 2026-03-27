from elasticai.explorer.hw_nas.search_space.build_model import DefaultModelBuilder
from elasticai.explorer.hw_nas.search_space.quantization import (
    FullPrecisionScheme,
    PTQFullyQuantizedInt8Scheme,
)
from elasticai.explorer.hw_nas.search_space.quantization_builder import (
    FullPrecisionBuilder,
    PTQFullyQuantizedInt8Builder,
)


from typing import Any


class PicoModelBuilder(DefaultModelBuilder):
    def get_supported_quantization_schemes(self) -> dict[str, Any]:
        return {
            PTQFullyQuantizedInt8Scheme.name(): PTQFullyQuantizedInt8Builder,
            FullPrecisionScheme.name(): FullPrecisionBuilder,
        }
