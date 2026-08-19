from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

from python_on_whales import docker

from elasticai.explorer.platforms.deployment.libtorch_installer import PiModel, setup_docker_libtorch

from settings import ROOT_DIR


@dataclass
class CompilerParams:
    image_name: str = "cross"
    library_path: Path = Path("./code/libtorch")
    path_to_dockerfile: Path = ROOT_DIR / "docker" / "Dockerfile.pibase"
    build_context: Path = ROOT_DIR / "docker"
    pi_model: Optional[str] = None


class Compiler(ABC):
    @abstractmethod
    def __init__(self, compiler_params: CompilerParams):
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
    def __init__(self, compiler_params: CompilerParams):
        self.logger = logging.getLogger("RPICompiler")
        self.image_name: str = compiler_params.image_name  # "cross"
        self.path_to_dockerfile: Path = Path(compiler_params.path_to_dockerfile)
        self.context_path: Path = Path(compiler_params.build_context)
        self.libtorch_path: Path = Path(compiler_params.library_path)
        self.pi_model: Optional[str] = compiler_params.pi_model
        if not self.is_setup():
            self.setup()
        self._ensure_libtorch()

    def is_setup(self) -> bool:
        return bool(docker.images(self.image_name))

    # todo: docker image in docker_registry
    def setup(self) -> None:
        self.logger.info("Crosscompiler has not been Setup. Setup Crosscompiler...")
        docker.build(
            self.context_path, file=self.path_to_dockerfile, tags=self.image_name
        )
        self.logger.debug("Crosscompiler available now.")

    def _is_libtorch_available(self) -> bool:
        libtorch_dir = (self.context_path / self.libtorch_path).resolve()
        if not libtorch_dir.exists():
            return False
        real_contents = [e for e in libtorch_dir.iterdir() if not e.name.startswith("._")]
        return bool(real_contents)

    def _ensure_libtorch(self) -> None:
        if self._is_libtorch_available():
            return
        if self.pi_model is None:
            self.logger.warning(
                f"libtorch not found in build context ({self.context_path / self.libtorch_path}) and no pi_model set in CompilerParams",
            )
            return

        self.logger.info("libtorch not found — downloading for %s...", self.pi_model)
        setup_docker_libtorch(PiModel(self.pi_model))

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

    def __init__(self, compiler_params: CompilerParams):
        self.logger = logging.getLogger("PicoCompiler")
        self.context_path: Path = Path(compiler_params.build_context)
        self.image_name: str = compiler_params.image_name
        self.path_to_dockerfile: Path = Path(compiler_params.path_to_dockerfile)
        self.context_path: Path = Path(compiler_params.build_context)
        self.cross_compiler_path: Path = Path(compiler_params.library_path)
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
