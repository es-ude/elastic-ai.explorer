from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tarfile

from python_on_whales import docker
from settings import ROOT_DIR
from elasticai.explorer.utils import synthesis_utils


@dataclass
class DockerParams:
    image_name: str = "cross"
    library_path: Path = Path("./code/libtorch")
    path_to_dockerfile: Path = ROOT_DIR / "docker" / "Dockerfile.picross"
    build_context: Path = ROOT_DIR / "docker"


@dataclass
class VivadoParams:
    remote_working_dir: str
    host: str
    ssh_user: str = "vivado"
    target_platform_name: str = ""


class Compiler(ABC):
    def __init__(self, compiler_params: DockerParams | VivadoParams):
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
    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path | None:
        pass


class RPICompiler(Compiler):
    def __init__(self, compiler_params: DockerParams):
        super().__init__(compiler_params)
        self.compiler_params = compiler_params
        self.image_name: str = compiler_params.image_name  # "cross"
        self.path_to_dockerfile: Path = Path(compiler_params.path_to_dockerfile)
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
            file=self.compiler_params.path_to_dockerfile,
            tags=self.compiler_params.image_name,
        )
        self.logger.debug("Crosscompiler available now.")

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path | None:
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

    def __init__(self, compiler_params: DockerParams):
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
            file=self.compiler_params.path_to_dockerfile,
            build_args={
                "CROSS_COMPILER_PATH": str(self.compiler_params.library_path),
            },
        )

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path | None:
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


class ENv5Compiler(Compiler):
    def __init__(self, compiler_params: VivadoParams):
        super().__init__(compiler_params=compiler_params)
        self.compiler_params = compiler_params

    def setup(self) -> None:
        pass

    def is_setup(self) -> bool:
        return True

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path | None:

        if self.compiler_params.target_platform_name == "env5_s50":
            target = synthesis_utils.TargetPlatforms.env5_s50
        elif self.compiler_params.target_platform_name == "env5_s15":
            target = synthesis_utils.TargetPlatforms.env5_s15
        else:
            err = ValueError(
                f"The platform {self.compiler_params.target_platform_name} is not supported by {self}"
            )
            self.logger.error(err)
            raise err

        try:
            path_to_bin_file = synthesis_utils.run_vhdl_synthesis(
                src_dir=source,
                remote_working_dir=self.compiler_params.remote_working_dir,
                host=self.compiler_params.host,
                ssh_user=self.compiler_params.ssh_user,
                target=target,
            )
        except Exception as e:
            self.logger.error(e)
            self.logger.info(f"The code from source {source}, could not be compiled!")

            path_to_bin_file = None

        tar = tarfile.open(str(output_dir) + "/vivado_run_results.tar.gz")
        tar.extractall(output_dir)
        tar.close()
        try:
            os.remove(str(output_dir) + "/vivado_run_results.tar.gz")
        except:
            pass

        return path_to_bin_file
