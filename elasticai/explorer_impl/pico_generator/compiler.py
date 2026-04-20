from elasticai.explorer.generator.deployment.compiler import Compiler, CompilerParams


from python_on_whales import docker


from pathlib import Path


class PicoCompiler(Compiler):

    def __init__(self, compiler_params: CompilerParams):
        super().__init__(compiler_params)
        self.compiler_params = compiler_params

       
        if not self.is_setup():
            self.setup()

    def is_setup(self) -> bool:
        return bool(docker.images(self.compiler_params.base_image_name))

    def setup(self) -> None:

        docker.build(
            context_path=self.compiler_params.build_context,
            tags=self.compiler_params.base_image_name,
            file=self.compiler_params.base_dockerfile_path,
            build_args={
                "CROSS_COMPILER_PATH": str(self.compiler_params.library_path),
                "PICO_TYPE": self.compiler_params.additional_params.get("platform_type", "rp2040"),
            },
        )

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path:
        context_path = self.compiler_params.build_context
        if not self.compiler_params.additional_params.get("platform_type"):
            self.logger.warning(
                    "No platform type given in additional parameters -> Pico Type defaults to RP2040."
                )
        docker.build(
            context_path=context_path,
            tags="pico-builder",
            output={
                "type": "local",
                "dest": str(context_path / "bin"),
            },
            file=self.compiler_params.cross_dockerfile_path,
            build_args={
                "BASE_IMAGE": self.compiler_params.base_image_name,
                "SOURCE_NAME": source.stem,
                "PATH_TO_SOURCE": str(source.as_posix()),
                "CROSS_COMPILER_PATH": str(self.compiler_params.library_path.as_posix()),
                "PICO_TYPE": self.compiler_params.additional_params.get("platform_type", "rp2040"),
            },
        )
        return context_path / "bin" / (source.stem + ".uf2")
