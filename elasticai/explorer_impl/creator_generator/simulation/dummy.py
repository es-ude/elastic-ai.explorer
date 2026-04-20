from pathlib import Path
from typing import Any

from elasticai.explorer.generator.deployment.compiler import Compiler
from elasticai.explorer.generator.deployment.device_communication import SerialHost


class DummyHost(SerialHost):
    def flash(self, local_path: Path):
        pass

    def receive(self, **kwargs) -> Any:
        pass

    def send_data_bytes(self, sample: bytearray, num_bytes_outputs: int) -> bytearray:
        return bytearray([])


class DummyCompiler(Compiler):
    def is_setup(self) -> bool:
        return True

    def setup(self) -> None:
        pass

    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path:
        return Path("")
