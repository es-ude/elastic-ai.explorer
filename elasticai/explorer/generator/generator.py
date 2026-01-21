from elasticai.explorer.generator.model_builder.model_builder import (
    DefaultModelBuilder,
    ModelBuilder,
)
from elasticai.explorer.generator.deployment.compiler import Compiler
from elasticai.explorer.generator.deployment.device_communication import Host
from elasticai.explorer.generator.deployment.hw_manager import HWManager
from elasticai.explorer.generator.model_compiler.model_compiler import ModelCompiler


from dataclasses import dataclass
from typing import Type


@dataclass
class Generator:
    hw_platform_name: str
    info: str
    model_compiler: Type[ModelCompiler]
    platform_manager: Type[HWManager]
    communication_protocol: Type[Host]
    compiler: Type[Compiler]
    model_builder: Type[ModelBuilder] = DefaultModelBuilder
