import math
import tomllib

import pytest
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import HWPlatform, KnowledgeRepository
from elasticai.explorer.platforms.deployment.compiler import (
    CompilerParams,
    PicoCompiler,
)
from elasticai.explorer.platforms.deployment.hw_manager import (
    Metric,
    PicoHWManager,
)
from elasticai.explorer.platforms.generator.generator import PicoGenerator
from elasticai.explorer.platforms.deployment.device_communication import (
    PicoHost,
    SerialParams,
)
from pathlib import Path

from elasticai.explorer.training import data
from elasticai.explorer.utils.data_utils import setup_mnist_for_cpp
from settings import DOCKER_CONTEXT_DIR, ROOT_DIR
from torchvision import transforms


class TestPicoDeploymentAndMeasurement:
    def setup_class(self):
        with open("./tests/system_tests/system_test_settings.toml", "rb") as f:
            self.config = tomllib.load(f)

        self.compiler_params = CompilerParams(
            library_path=Path("./code/pico_crosscompiler"),
            image_name="picobase",
            build_context=DOCKER_CONTEXT_DIR,
            path_to_dockerfile=ROOT_DIR / "docker/Dockerfile.picobase",
        )  # <-- Configure this only if necessary.
        knowledge_repository = KnowledgeRepository()
        knowledge_repository.register_hw_platform(
            HWPlatform(
                "pico",
                "Pico with RP2040 MCU and 2MB control memory",
                PicoGenerator,
                PicoHWManager,
                PicoHost,
                PicoCompiler,
            )
        )
        knowledge_repository.register_hw_platform(
            HWPlatform(
                "pico2w",
                "Pico2w with RP2350 MCU and 4MB control memory",
                PicoGenerator,
                PicoHWManager,
                PicoHost,
                PicoCompiler,
            )
        )

        self.pico_explorer = Explorer(knowledge_repository)
        self.pico_explorer.experiment_dir = ROOT_DIR / Path(
            "tests/system_tests/test_experiment"
        )
        self.pico_explorer._model_dir = ROOT_DIR / Path("tests/system_tests/samples")
        self.model_name = "ts_model_0.tflite"

        self.metric_to_source = {
            Metric.ACCURACY: Path(
                "code/pico_crosscompiler/measure_accuracy"
            ),  # test relative path
            Metric.LATENCY: (
                DOCKER_CONTEXT_DIR / Path("code/pico_crosscompiler/measure_latency")
            ),  # test absolute path
        }
        transf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        path_to_dataset = ROOT_DIR / "data/mnist"
        path_to_deployable_dataset = ROOT_DIR / "data/cpp-mnist"

        setup_mnist_for_cpp(
            root_dir_mnist=path_to_dataset,
            root_dir_cpp_mnist=path_to_deployable_dataset,
            transf=transf,
        )

        self.dataset_spec = data.DatasetSpecification(
            dataset=data.MNISTWrapper(root=path_to_dataset, transform=transf),
            deployable_dataset_path=path_to_deployable_dataset,
        )

    @pytest.mark.hardware
    @pytest.mark.parametrize(
        ("image_name", "docker_file", "DEVICE_PATH_KEY"),
        [
            ("picobase", "docker/Dockerfile.picobase", "PICO_DEVICE_PATH"),
            ("pico2wbase", "docker/Dockerfile.pico2wbase", "PICO2W_DEVICE_PATH"),
        ],
    )
    def test_pico_accuracy_measurement(self, image_name, docker_file, DEVICE_PATH_KEY):
        serial_params = SerialParams(self.config[DEVICE_PATH_KEY])
        compiler_params = CompilerParams(
            library_path=Path("./code/pico_crosscompiler"),
            image_name=image_name,
            build_context=DOCKER_CONTEXT_DIR,
            path_to_dockerfile=ROOT_DIR / docker_file,
        )  # <-- Configure this only if necessary.
        self.pico_explorer.choose_target_hw("pico", compiler_params, serial_params)
        self.pico_explorer.hw_setup_on_target(self.metric_to_source, self.dataset_spec)

        assert math.isclose(
            self.pico_explorer.run_measurement(
                Metric.ACCURACY, model_name=self.model_name
            )["Accuracy"]["value"],
            78.516,
            abs_tol=0.01,
        )

    @pytest.mark.hardware
    @pytest.mark.parametrize(
        ("image_name", "docker_file", "DEVICE_PATH_KEY"),
        [
            ("picobase", "docker/Dockerfile.picobase", "PICO_DEVICE_PATH"),
            ("pico2wbase", "docker/Dockerfile.pico2wbase", "PICO2W_DEVICE_PATH"),
        ],
    )
    def test_pico_latency_measurement(self, image_name, docker_file, DEVICE_PATH_KEY):
        serial_params = SerialParams(self.config[DEVICE_PATH_KEY])
        compiler_params = CompilerParams(
            library_path=Path("./code/pico_crosscompiler"),
            image_name=image_name,
            build_context=DOCKER_CONTEXT_DIR,
            path_to_dockerfile=ROOT_DIR / docker_file,
        )  # <-- Configure this only if necessary.
        self.pico_explorer.choose_target_hw("pico", compiler_params, serial_params)
        self.pico_explorer.hw_setup_on_target(self.metric_to_source, self.dataset_spec)

        assert (
            type(
                self.pico_explorer.run_measurement(
                    Metric.LATENCY,
                    model_name=self.model_name,
                )["Latency"]["value"]
            )
            == int
        )
