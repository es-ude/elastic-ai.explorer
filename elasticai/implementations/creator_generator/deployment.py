import logging
import os
from pathlib import Path
import tarfile
from typing import Any

import elasticai.creator.nn as creator_nn
from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.vhdl.system_integrations.firmware_env5 import FirmwareENv5

from elasticai.creator.nn import fixed_point
from torch import nn
import torch

from elasticai.explorer.generator.deployment.compiler import Compiler, VivadoParams
from elasticai.explorer.generator.deployment.device_communication import (
    SerialHost,
    SerialParams,
)
from elasticai.explorer.generator.deployment.hw_manager import HWManager
from elasticai.explorer.generator.model_translator.model_translator import (
    ModelTranslator,
)


from elasticai.explorer.hw_nas.search_space.quantization import (
    CreatorFixedPointScheme,
    QuantizationScheme,
)
from torch.utils.data import DataLoader, random_split

# TODO make these creator imports optional
from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)

from elasticai.explorer.training.data import DatasetSpecification
from elasticai.implementations.creator_generator import synthesis_utils
import serial

from elasticai.runtime.env5.usb import UserRemoteControl, get_env5_port


class CreatorModelTranslator(ModelTranslator):
    def __init__(self) -> None:
        self.logger = logging.getLogger(
            "explorer.generator.model_compiler.model_compiler.CreatorModelCompiler"
        )
        self.skeleton_id = [2 for i in range(16)]

    def translate(
        self,
        model: nn.Module,
        output_path: Path,
        sample: torch.Tensor,
        quantization_scheme: QuantizationScheme = CreatorFixedPointScheme(),
    ):
        destination = OnDiskPath(str(output_path), parent="")
        features_in = len(sample)
        features_out = len(model(sample))
        if not isinstance(model, creator_nn.Sequential):
            err = TypeError(
                f"{type(model)} is not supported by the CreatorModelTranslator, best to build models with the CreatorModelBuilder!"
            )
            self.logger.error(err)
            raise err

        my_design = model.create_design("myNetwork")

        my_design.save_to(destination.create_subpath("srcs"))

        firmware = FirmwareENv5(
            network=my_design,
            x_num_values=features_in,
            y_num_values=features_out,
            id=self.skeleton_id,
            skeleton_version="v2",
        )
        firmware.save_to(destination)

    def get_supported_layers(self) -> tuple[type] | None:
        return (fixed_point.Linear,)

    def get_supported_quantization_schemes(
        self,
    ) -> tuple[type[QuantizationScheme]] | None:
        return (CreatorFixedPointScheme,)


class ENv5Compiler(Compiler):
    def __init__(self, compiler_params: VivadoParams):
        super().__init__(compiler_params=compiler_params)
        self.compiler_params = compiler_params

    def setup(self) -> None:
        pass

    def is_setup(self) -> bool:
        return True

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path | None:

        if self.compiler_params.target_platform_name == "env5_s50":
            target = synthesis_utils.TargetPlatforms.env5_s50
        elif self.compiler_params.target_platform_name == "env5_s15":
            target = synthesis_utils.TargetPlatforms.env5_s15
        else:
            err = ValueError(
                f"The platform {self.compiler_params.target_platform_name} is not supported by {self}"
            )
            self.logger.error(err)
            raise err

        try:
            path_to_bin_file = synthesis_utils.run_vhdl_synthesis(
                src_dir=source,
                remote_working_dir=self.compiler_params.remote_working_dir,
                host=self.compiler_params.host,
                ssh_user=self.compiler_params.ssh_user,
                target=target,
            )
        except Exception as e:
            self.logger.error(e)
            self.logger.info(f"The code from source {source}, could not be compiled!")

            path_to_bin_file = None

        tar = tarfile.open(str(output_dir) + "/vivado_run_results.tar.gz")
        tar.extractall(output_dir)
        tar.close()
        try:
            os.remove(str(output_dir) + "/vivado_run_results.tar.gz")
        except:
            pass

        return path_to_bin_file


class ENv5Host(SerialHost):
    def __init__(self, params: SerialParams):
        super().__init__(params=params)
        self.logger = logging.getLogger(
            "explorer.generator.deployment.device_communication.ENv5Host"
        )
        self.flash_start_address = 0
        self._ser = None

    def _get_connection(self) -> serial.Serial:
        if not self._ser:
            self._ser = serial.Serial(
                get_env5_port(), baudrate=self.BAUD_RATE, timeout=1
            )
        return self._ser

    def flash(self, local_path: Path):
        skeleton_id = [2 for i in range(16)]
        skeleton_id_as_bytearray = bytearray()
        for x in skeleton_id:
            skeleton_id_as_bytearray.extend(
                x.to_bytes(length=1, byteorder="little", signed=False)
            )

        ser = self._get_connection()
        self._urc = UserRemoteControl(device=ser)
        self._urc.send_and_deploy_model(
            local_path, self.flash_start_address, skeleton_id_as_bytearray
        )
        self._urc.fpga_leds(True, False, False, False)
        skeleton_id_on_device = bytearray(self._urc._enV5RCP.read_skeleton_id())

        if skeleton_id_on_device == skeleton_id_as_bytearray:
            self.logger.info("The byte stream has been written correctly to the ENv5.")
        else:
            self.logger.warning(
                f"The byte stream hasn't been written correctly to the ENv5. Verification bytes are not equal: {skeleton_id_on_device} != {skeleton_id_as_bytearray}!"
            )

    def send_data_bytes(self, sample: bytearray, num_bytes_outputs: int) -> bytearray:
        if self._urc:
            raw_result = self._urc.inference_with_data(sample, num_bytes_outputs)
            return raw_result

        with self._get_connection() as ser:
            urc = UserRemoteControl(device=ser)
            raw_result = urc.inference_with_data(sample, num_bytes_outputs)
            return raw_result

    def receive(self, **kwargs) -> Any:
        if self._urc:
            raw_result = self._urc.read_data_from_flash(
                kwargs.get("flash_start_address", 0), kwargs.get("num_bytes", 0)
            )
            return raw_result


class ENv5HWManager(HWManager):
    def __init__(self, target: ENv5Host, compiler: Compiler):
        self.compiler = compiler
        self.target = target
        self.logger = logging.getLogger(
            "explorer.generator.deployment.hw_manager.ENv5HWManager"
        )
        self.logger.info("Initializing PI Hardware Manager...")
        super().__init__(target, compiler)

    def prepare_dataset(
        self,
        dataset_spec: DatasetSpecification,
        quantization_scheme: QuantizationScheme,
    ):
        super().prepare_dataset(dataset_spec, quantization_scheme)
        if type(quantization_scheme) is not CreatorFixedPointScheme:
            err = TypeError("Env5 only Supports CreatorFixedPointScheme")
            self.logger.error(err)
            raise err

        self.frac_bits = quantization_scheme.frac_bits
        self.total_bits = quantization_scheme.total_bits
        self.dataset_spec = dataset_spec
        fxp_params = FxpParams(
            total_bits=quantization_scheme.total_bits,
            frac_bits=quantization_scheme.frac_bits,
            signed=quantization_scheme.signed,
        )
        fxp_conf = FxpArithmetic(fxp_params)
        self.dataset = self.dataset_spec.dataset
        _, test_subset, _ = random_split(
            self.dataset,
            dataset_spec.train_val_test_ratio,
        )
        self.batch_size = 64
        self.test_loader = DataLoader(
            test_subset, batch_size=self.batch_size, shuffle=dataset_spec.shuffle
        )

    def prepare_model(self, path_to_model: Path):

        self.path_to_executable = self.compiler.compile_code(
            path_to_model, path_to_model.parent
        )

        if self.path_to_executable:
            self.target.flash(local_path=self.path_to_executable)


# TODO add simulation for latency and accuracy measurement
