from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tarfile

from python_on_whales import docker
from settings import ROOT_DIR
from elasticai.explorer.generator.reflection import Reflective
from elasticai.explorer.hw_nas.search_space.quantization import (
    FixedPointInt8Scheme,
    QuantizationScheme,
)
from elasticai.explorer.utils import synthesis_utils

@dataclass
class CompilerParams:
    image_name: str = "cross"
    library_path: Path = Path("./code/libtorch")
    path_to_dockerfile: Path = ROOT_DIR / "docker" / "Dockerfile.picross"
    build_context: Path = ROOT_DIR / "docker"


class Compiler(ABC, Reflective):
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
            self.context_path, file=self.path_to_dockerfile, tags=self.image_name
        )
        self.logger.debug("Crosscompiler available now.")

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path:
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
        super().__init__(compiler_params)
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

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path:

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


class ENv5Compiler(Compiler):
    def setup(self) -> None:
        pass

    def is_setup(self) -> bool:
        return True

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path:

        if self.deploy_cfg.target_platform_name == "env5_s50":
            target = synthesis_utils.TargetPlatforms.env5_s50
        elif self.deploy_cfg.target_platform_name == "env5_s15":
            target = synthesis_utils.TargetPlatforms.env5_s15
        else:
            err = ValueError(
                f"The platform {self.deploy_cfg.target_platform_name} is not supported by {self}"
            )
            self.logger.error(err)
            raise err

        path_to_bin_file = synthesis_utils.run_vhdl_synthesis(
            src_dir=source,
            remote_working_dir=self.deploy_cfg.vivado_build_server.remote_working_dir,
            host=self.deploy_cfg.vivado_build_server.host,
            ssh_user=self.deploy_cfg.vivado_build_server.ssh_user,
            target=target,
        )

        tar = tarfile.open(str(output_dir) + "/vivado_run_results.tar.gz")
        tar.extractall(output_dir)
        tar.close()
        try:
            os.remove(str(output_dir) + "/vivado_run_results.tar.gz")
        except:
            pass

        return path_to_bin_file

    def get_supported_quantization_schemes(
        self,
    ) -> tuple[type[QuantizationScheme]] | None:
        return (FixedPointInt8Scheme,)
