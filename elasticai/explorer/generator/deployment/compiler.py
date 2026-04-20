from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from pathlib import Path


@dataclass
class CompilerParams:
    base_dockerfile_path: Path  # The path to the base dockerfile. The base dockerfile gives instruction on how to build the base image.
    cross_dockerfile_path: Path 
    build_context: Path  # The absolute path to the build context. For Docker this should be containing sources and Dockerfiles.
    base_image_name: str = "pibase"
    library_path: Path = Path(
        "./code/libtorch"
    )  # This should be relative to the build context with a leading "./"
    additional_params: dict = field(default_factory=lambda: {})

class Compiler(ABC):
    def __init__(self, compiler_params: CompilerParams):
        self.compiler_params = compiler_params
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self.logger = logging.getLogger(logger_name)

    @abstractmethod
    def is_setup(self) -> bool:
        pass

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path:
        pass
