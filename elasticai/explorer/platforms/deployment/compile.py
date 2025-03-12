from python_on_whales import docker

from elasticai.explorer.platforms.deployment.manager import CONTEXT_PATH


class Compiler:
    def __init__(self, config):
        self.tag: str = config.compiler_tag  # "cross"
        self.path_to_dockerfile = config.path_to_dockerfile  # CONTEXT_PATH / "Dockerfile.picross"
        if not self.is_setup():
            self.setup()

    def is_setup(self) -> bool:
        return docker.images(self.tag)

    # todo: docker image in docker_registry
    def setup(self) -> None:
        self.logger.info("Crosscompiler has not been Setup. Setup Crosscompiler...")
        docker.build(
            CONTEXT_PATH, file=self.path_to_dockerfile, tags=self.tag
        )
        self.logger.debug("Crosscompiler available now.")

    def compile_code(self, path_to_program: str):
        docker.build(
            CONTEXT_PATH,
            file=CONTEXT_PATH / "Dockerfile.loader",
            output={"type": "local", "dest": CONTEXT_PATH / "bin"},
        )
        self.logger.info("Compilation finished. Program available in %s", CONTEXT_PATH / "bin")
