import logging
from pathlib import Path

from python_on_whales import docker

from elasticai.explorer.config import DeploymentConfig


class Compiler:
    def __init__(self, deploy_cfg: DeploymentConfig):
        self.logger = logging.getLogger("Compiler")
        self.tag: str = deploy_cfg.docker.compiler_tag  # "cross"
        self.path_to_dockerfile: Path = Path(
            deploy_cfg.docker.path_to_dockerfile
        )  # CONTEXT_PATH / "Dockerfile.picross"
        self.context_path: Path = Path(deploy_cfg.docker.build_context)
        self.libtorch_path: Path = Path(deploy_cfg.docker.compiled_library_path)
        if not self.is_setup():
            self.setup()

    def is_setup(self) -> bool:
        return bool(docker.images(self.tag))

    # todo: docker image in docker_registry
    def setup(self) -> None:
        self.logger.info("Crosscompiler has not been Setup. Setup Crosscompiler...")
        docker.build(self.context_path, file=self.path_to_dockerfile, tags=self.tag)
        self.logger.debug("Crosscompiler available now.")

    def compile_code(self, name_of_executable: str, sourcecode_filename: str):
        docker.build(
            self.context_path,
            file=self.context_path / "Dockerfile.loader",
            output={"type": "local", "dest": str(self.context_path / "bin")},
            build_args={
                "NAME_OF_EXECUTABLE": name_of_executable,
                "PROGRAM_CODE": sourcecode_filename,
                "HOST_LIBTORCH_PATH": str(self.libtorch_path),
            },
        )
        path_to_executable = self.context_path / "bin" / name_of_executable
        self.logger.info(
            "Compilation finished. Program available in %s", path_to_executable
        )
        return path_to_executable
