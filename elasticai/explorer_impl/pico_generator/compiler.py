from elasticai.explorer.generator.deployment.compiler import Compiler, CompilerParams


from python_on_whales import docker


from pathlib import Path


class PicoCompiler(Compiler):

    def __init__(self, compiler_params: CompilerParams, **kwargs):
        super().__init__(compiler_params, **kwargs)
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
