from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import serial
from fabric import Connection


@dataclass
class SSHParams:
    hostname: str
    username: str


@dataclass
class SerialParams:
    device_path: Path
    serial_port: str = "/dev/ttyACM0"
    baud_rate: int = 115200


class SSHException(Exception):
    pass


class Host(ABC):
    @abstractmethod
    def __init__(self, params): ...

    @abstractmethod
    def _get_connection(self) -> Any: ...


class SSHHost(Host):
    def __init__(self, params: SSHParams):
        self.hostname = params.hostname
        self.username = params.username
        self.logger = logging.getLogger(
            "explorer.generator.deployment.device_communication.SSHHost"
        )

    def _get_connection(self):
        return Connection(host=self.hostname, user=self.username)

    @abstractmethod
    def put_file(self, local_path: Path, remote_path: str | None) -> str: ...
    @abstractmethod
    def run_command(self, command: str) -> str: ...


class SerialHost(Host):

    def __init__(self, params: SerialParams):
        self.BAUD_RATE = params.baud_rate
        self.host_name = params.device_path
        self.logger = logging.getLogger(
            "explorer.generator.deployment.device_communication.SerialHost"
        )
        self.serial_port = params.serial_port
        self.timeout_s = 40

    def _get_connection(self) -> serial.Serial:
        return serial.Serial(self.serial_port, self.BAUD_RATE, timeout=1)

    @abstractmethod
    def flash(self, local_path: Path): ...
    @abstractmethod
    def receive(self, **kwargs) -> Any: ...
    @abstractmethod
    def send_data_bytes(
        self, sample: bytearray, num_bytes_outputs: int
    ) -> bytearray: ...
