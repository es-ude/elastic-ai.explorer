import logging
from pathlib import Path

from python_on_whales import docker


class Compiler:
    def __init__(self, config):
        self.logger = logging.getLogger("Compiler")
        self.tag: str = config.compiler_tag  # "cross"
        self.path_to_dockerfile: Path = Path(config.path_to_dockerfile)  # CONTEXT_PATH / "Dockerfile.picross"
        self.context_path: Path = Path(config.build_context)
        if not self.is_setup():
            self.setup()

    def is_setup(self) -> bool:
        return docker.images(self.tag)

    # todo: docker image in docker_registry
    def setup(self) -> None:
        self.logger.info("Crosscompiler has not been Setup. Setup Crosscompiler...")
        docker.build(
            self.context_path, file=self.path_to_dockerfile, tags=self.tag
        )
        self.logger.debug("Crosscompiler available now.")

    def compile_code(self, name_of_executable: str, sourcecode_filename: str):
        docker.build(
            self.context_path,
            file=self.context_path / "Dockerfile.loader",
            output={"type": "local", "dest": self.context_path / "bin"},
            build_args={"NAME_OF_EXECUTABLE": name_of_executable, "PROGRAM_CODE": sourcecode_filename}
        )
        path_to_executable = self.context_path / "bin" / name_of_executable
        self.logger.info("Compilation finished. Program available in %s", path_to_executable)
        return path_to_executable
