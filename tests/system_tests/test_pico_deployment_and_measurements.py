import math
import tomllib

import pytest
from elasticai.explorer.explorer import Explorer
from elasticai.explorer_impl.pico_generator.hw_manager import PicoHWManager
from elasticai.explorer_impl.pico_generator.host import PicoHost
from elasticai.explorer_impl.pico_generator.compiler import PicoCompiler
from elasticai.explorer.generator.generator import Generator
from elasticai.explorer.generator_registry import GeneratorRegistry
from elasticai.explorer.generator.deployment.compiler import (
    CompilerParams,
)
from elasticai.explorer.generator.deployment.hw_manager import (
    Metric,
)
from elasticai.explorer_impl.pico_generator.model_translator import (
    TFliteModelTranslator,
)
from elasticai.explorer.generator.deployment.device_communication import (
    SerialParams,
)
from pathlib import Path

from elasticai.explorer.training import data
from elasticai.explorer_impl.pico_generator.utils import (
    prepare_image_dataset_for_cpp,
)
from settings import DOCKER_CONTEXT_DIR, ROOT_DIR
from torchvision import datasets, transforms


class TestPicoDeploymentAndMeasurement:
    def setup_class(self):
        with open("./tests/system_tests/system_test_settings.toml", "rb") as f:
            self.config = tomllib.load(f)

        self.compiler_params = CompilerParams(
            library_path=Path("./code/pico_crosscompiler"),
            base_image_name="picobase",
            build_context=DOCKER_CONTEXT_DIR,
            base_dockerfile_path=ROOT_DIR / "docker/Dockerfile.picobase",
            cross_dockerfile_path=ROOT_DIR / "docker/Dockerfile.picocross"
        )  # <-- Configure this only if necessary.
        generator_registry = GeneratorRegistry()
        generator_registry.register_generator(
            Generator(
                "pico",
                "Pico with RP2040 MCU and 2MB control memory",
                TFliteModelTranslator,
                PicoHWManager,
                PicoHost,
                PicoCompiler,
            )
        )
        self.pico_explorer = Explorer(
            generator_registry, ROOT_DIR / Path("tests/system_tests"), "test_experiment"
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

        dataset = datasets.MNIST(
            root=path_to_dataset, train=False, download=True, transform=transf
        )

        prepare_image_dataset_for_cpp(
            dataset,
            output_dir=path_to_deployable_dataset,
            num_samples=256,
            dtype="float",
            flatten=True,
        )

        self.dataset_spec = data.DatasetSpecification(
            dataset=data.MNISTWrapper(root=path_to_dataset, transform=transf),
            deployable_dataset_path=path_to_deployable_dataset,
        )

    @pytest.mark.hardware
    @pytest.mark.parametrize(
        ("image_name", "DEVICE_PATH_KEY", "platform_type"),
        [
            (
                "picobase",
                "PICO_DEVICE_PATH",
                "rp2040",
            ),
            (
                "pico2base",
                "PICO2_DEVICE_PATH",
                "rp2350",
            ),
        ],
    )
    def test_pico_accuracy_measurement(
        self, image_name, DEVICE_PATH_KEY, platform_type
    ):
        serial_params = SerialParams(self.config[DEVICE_PATH_KEY])
        compiler_params = CompilerParams(
            library_path=Path("./code/pico_crosscompiler"),
            base_image_name=image_name,
            build_context=DOCKER_CONTEXT_DIR,
            base_dockerfile_path=ROOT_DIR / "docker/Dockerfile.picobase",
            cross_dockerfile_path=ROOT_DIR / "docker/Dockerfile.picocross",
            additional_params={"platform_type": platform_type},
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
        ("image_name", "DEVICE_PATH_KEY", "platform_type"),
        [
            (
                "picobase",
                "PICO_DEVICE_PATH",
                "rp2040",
            ),
            (
                "pico2base",
                "PICO2_DEVICE_PATH",
                "rp2350",
            ),
        ],
    )
    def test_pico_latency_measurement(
        self, image_name,DEVICE_PATH_KEY, platform_type
    ):
        serial_params = SerialParams(self.config[DEVICE_PATH_KEY])
        compiler_params = CompilerParams(
            library_path=Path("./code/pico_crosscompiler"),
            base_image_name=image_name,
            build_context=DOCKER_CONTEXT_DIR,
            base_dockerfile_path=ROOT_DIR / "docker/Dockerfile.picobase",
            cross_dockerfile_path=ROOT_DIR / "docker/Dockerfile.picocross",
            additional_params={"platform_type": platform_type},
        )   # <-- Configure this only if necessary.
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
