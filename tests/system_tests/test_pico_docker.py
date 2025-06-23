import os
from python_on_whales import docker

from settings import ROOT_DIR


class TestPicoCompile:
    def setup_method(self):
        self.context_path = ROOT_DIR / "docker"

    def test_pico_docker(self):

        output_dir = os.path.abspath("docker/bin")
        os.makedirs(output_dir, exist_ok=True)

        docker.build(
            context_path=".",
            tags="picobase",
            file="docker/Dockerfile.picobase",
            build_args={
                "CROSS_COMPILER_PATH": "docker/code/pico_crosscompiler",
            },
        )

        docker.build(
            context_path=".",
            tags="pico-builder",
            output={"type": "local", "dest": str(self.context_path / "bin")},
            file="docker/Dockerfile.picocross",
            build_args={
                "NAME_OF_EXECUTABLE": "app_full_precision.uf2",
                "SOURCE_CODE": "app_full_precision",
                "CROSS_COMPILER_PATH": "docker/code/pico_crosscompiler",
            },
        )

    def teardown_method(self):
        pass
