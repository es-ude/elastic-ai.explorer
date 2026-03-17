from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
from socket import error as socket_error
import time
from typing import Any

import serial
from fabric import Connection
from paramiko.ssh_exception import AuthenticationException



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


class RPiHost(SSHHost):

    def __init__(self, params: SSHParams):
        super().__init__(params=params)
        self.logger = logging.getLogger(
            "explorer.generator.deployment.device_communication.RPiHost"
        )

    def run_command(self, command: str) -> str:
        try:
            with self._get_connection() as conn:
                self.logger.info(
                    "Install program on target. Hostname: %s - User: %s",
                    conn.host,
                    conn.user,
                )
                result = conn.run(command, warn=True, hide=True)
        except (socket_error, AuthenticationException) as exc:
            self._raise_authentication_err(exc)

        if result.failed:
            raise SSHException(
                "The command `{0}` on host {1} failed with the error: "
                "{2}".format(command, self.hostname, str(result.stderr))
            )
        return result.stdout

    def put_file(self, local_path: Path, remote_path: str | None) -> str:
        try:
            with self._get_connection() as conn:
                conn.put(local_path, remote_path)
        except (socket_error, AuthenticationException) as exc:
            self._raise_authentication_err(exc)

        return ""

    def _raise_authentication_err(self, exc):
        raise SSHException(
            "SSH: could not connect to {host} "
            "(username: {user}): {exc}".format(
                host=self.hostname, user=self.username, exc=exc
            )
        )


class PicoHost(SerialHost):
    def __init__(self, params: SerialParams):
        super().__init__(params=params)
        self.logger = logging.getLogger(
            "explorer.generator.deployment.device_communication.PicoHost"
        )

    def flash(self, local_path: Path):
        time_passed = 0
        sleep_interval = 0.5
        self.logger.info("Wait for pico to deploy...")
        while not os.path.isdir(self.host_name):
            time.sleep(sleep_interval)
            time_passed = time_passed + sleep_interval
            if time_passed > self.timeout_s:
                time.sleep(4)
                self.logger.error("Timeout on Pico-Communication")
                self.logger.info("Manual Reboot necessary")

        shutil.copyfile(
            local_path,
            Path(self.host_name) / Path(local_path).name,
        )

    def send_data_bytes(self, sample: bytearray, num_bytes_outputs: int) -> bytearray:
        with self._get_connection() as ser:
            return bytearray(ser.read_until().strip())

    def receive(self, **kwargs) -> str:
        self._wait_for_pico(self.serial_port)
        line = ""
        try:
            with self._get_connection() as ser:
                line = self._read_serial_once(ser)
        except serial.SerialException as e:
            self.logger.error("Error with serial communication!")
            raise e
        except PermissionError as e:
            self.logger.error(
                "Permission Error with serial communication! Probably you need to add the user to dialout and tty group."
            )
            raise e

        return line

    def _wait_for_pico(self, port):
        self.logger.info("Wait for pico answer on Port " + port + "...")
        time_passed = 0
        sleep_interval = 0.5
        while not os.path.exists(port):
            time.sleep(sleep_interval)
            time_passed = time_passed + sleep_interval
            if time_passed > self.timeout_s:
                self.logger.error("Timeout on Pico-Communication")
                exit(-1)

        time.sleep(1.0)

    def _read_serial_once(
        self,
        ser,
    ) -> str:
        last_line = ""
        start_time_s = time.time()
        time_passed_s = 0
        while True:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    last_line = line
            except serial.SerialException:
                break
            if time_passed_s > self.timeout_s:
                break
            time_passed_s = time.time() - start_time_s

        return last_line
