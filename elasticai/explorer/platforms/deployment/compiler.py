from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path

from python_on_whales import docker

from settings import ROOT_DIR


@dataclass
class DockerParams:
    image_name: str = "cross"
    library_path: Path = Path("./code/libtorch")
    path_to_dockerfile: Path = ROOT_DIR / "docker" / "Dockerfile.picross"
    build_context: Path = ROOT_DIR / "docker"


class Compiler(ABC):
    @abstractmethod
    def __init__(self, docker_params: DockerParams):
        pass

    @abstractmethod
    def is_setup(self) -> bool:
        pass

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def compile_code(self, source: Path) -> Path:
        pass


class RPICompiler(Compiler):
    def __init__(self, docker_params: DockerParams):
        self.logger = logging.getLogger("RPICompiler")
        self.image_name: str = docker_params.image_name  # "cross"
        self.path_to_dockerfile: Path = Path(docker_params.path_to_dockerfile)
        self.context_path: Path = Path(docker_params.build_context)
        self.libtorch_path: Path = Path(docker_params.library_path)
        if not self.is_setup():
            self.setup()

    def is_setup(self) -> bool:
        return bool(docker.images(self.image_name))

    # todo: docker image in docker_registry
    def setup(self) -> None:
        self.logger.info("Crosscompiler has not been Setup. Setup Crosscompiler...")
        docker.build(
            self.context_path, file=self.path_to_dockerfile, tags=self.image_name
        )
        self.logger.debug("Crosscompiler available now.")

    def compile_code(self, source: Path) -> Path:
        docker.build(
            self.context_path,
            file=self.context_path / "Dockerfile.picross",
            output={"type": "local", "dest": str(self.context_path / "bin")},
            build_args={
                "BASE_IMAGE": self.image_name,
                "NAME_OF_EXECUTABLE": source.stem,
                "PROGRAM_CODE": str(source),
                "HOST_LIBTORCH_PATH": str(self.libtorch_path),
            },
        )
        path_to_executable = self.context_path / "bin" / source.stem
        self.logger.info(
            "Compilation finished. Program available in %s", path_to_executable
        )
        return path_to_executable


class PicoCompiler(Compiler):

    def __init__(self, docker_params: DockerParams):
        self.logger = logging.getLogger("PicoCompiler")
        self.context_path: Path = Path(docker_params.build_context)
        self.image_name: str = docker_params.image_name
        self.path_to_dockerfile: Path = Path(docker_params.path_to_dockerfile)
        self.context_path: Path = Path(docker_params.build_context)
        self.cross_compiler_path: Path = Path(docker_params.library_path)
        if not self.is_setup():
            self.setup()

    def is_setup(self) -> bool:
        return bool(docker.images(self.image_name))

    def setup(self) -> None:
        docker.build(
            context_path=self.context_path,
            tags=self.image_name,
            file=self.path_to_dockerfile,
            build_args={
                "CROSS_COMPILER_PATH": str(self.cross_compiler_path),
            },
        )

    def compile_code(self, source: Path) -> Path:

        docker.build(
            context_path=self.context_path,
            tags="pico-builder",
            output={"type": "local", "dest": str(self.context_path / "bin")},
            file=self.context_path / "Dockerfile.picocross",
            build_args={
                "BASE_IMAGE": self.image_name,
                "SOURCE_NAME": source.stem,
                "PATH_TO_SOURCE": str(source),
                "CROSS_COMPILER_PATH": str(self.cross_compiler_path),
            },
        )
        return self.context_path / "bin" / (source.stem + ".uf2")
