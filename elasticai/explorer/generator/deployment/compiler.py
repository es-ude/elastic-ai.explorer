from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path


from python_on_whales import docker


@dataclass
class CompilerParams:
    base_dockerfile_path: Path  # The path to the base dockerfile. The base dockerfile gives instruction on how to build the base image.
    build_context: Path  # The absolute path to the build context. For Docker this should be containing sources and Dockerfiles.
    image_name: str = "pibase"
    library_path: Path = Path(
        "./code/libtorch"
    )  # This should be relative to the build context with a leading "./" 


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


class RPICompiler(Compiler):
    def __init__(self, compiler_params: CompilerParams):
        super().__init__(compiler_params)
        self.compiler_params = compiler_params
        self.image_name: str = compiler_params.image_name  # "cross"
        self.base_dockerfile_path: Path = Path(compiler_params.base_dockerfile_path)
        self.context_path: Path = Path(compiler_params.build_context)
        self.libtorch_path: Path = Path(compiler_params.library_path)
        if not self.is_setup():
            self.setup()

    def is_setup(self) -> bool:
        return bool(docker.images(self.image_name))

    # todo: docker image in docker_registry
    def setup(self) -> None:
        self.logger.info("Crosscompiler has not been Setup. Setup Crosscompiler...")
        docker.build(
            self.compiler_params.build_context,
            file=self.compiler_params.base_dockerfile_path,
            tags=self.compiler_params.image_name,
        )
        self.logger.debug("Crosscompiler available now.")

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path:
        context_path = self.compiler_params.build_context
        docker.build(
            context_path,
            file=context_path / "Dockerfile.picross",
            output={"type": "local", "dest": str(context_path / "bin")},
            build_args={
                "BASE_IMAGE": self.compiler_params.image_name,
                "NAME_OF_EXECUTABLE": source.stem,
                "PROGRAM_CODE": str(source),
                "HOST_LIBTORCH_PATH": str(self.compiler_params.library_path),
            },
        )
        path_to_executable = context_path / "bin" / source.stem
        self.logger.info(
            "Compilation finished. Program available in %s", path_to_executable
        )
        return path_to_executable


class PicoCompiler(Compiler):

    def __init__(self, compiler_params: CompilerParams):
        super().__init__(compiler_params)
        self.compiler_params = compiler_params
        if not self.is_setup():
            self.setup()

    def is_setup(self) -> bool:
        return bool(docker.images(self.compiler_params.image_name))

    def setup(self) -> None:

        docker.build(
            context_path=self.compiler_params.build_context,
            tags=self.compiler_params.image_name,
            file=self.compiler_params.base_dockerfile_path,
            build_args={
                "CROSS_COMPILER_PATH": str(self.compiler_params.library_path),
            },
        )

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path:
        context_path = self.compiler_params.build_context
        docker.build(
            context_path=context_path,
            tags="pico-builder",
            output={
                "type": "local",
                "dest": str(context_path / "bin"),
            },
            file=context_path / "Dockerfile.picocross",
            build_args={
                "BASE_IMAGE": self.compiler_params.image_name,
                "SOURCE_NAME": source.stem,
                "PATH_TO_SOURCE": str(source),
                "CROSS_COMPILER_PATH": str(self.compiler_params.library_path),
            },
        )
        return context_path / "bin" / (source.stem + ".uf2")
