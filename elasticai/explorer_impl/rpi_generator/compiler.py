from elasticai.explorer.generator.deployment.compiler import Compiler, CompilerParams


from python_on_whales import docker


from pathlib import Path


class RPICompiler(Compiler):
    def __init__(self, compiler_params: CompilerParams, **kwargs):
        super().__init__(compiler_params, **kwargs)
        self.compiler_params = compiler_params
        self.image_name: str = compiler_params.image_name
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
