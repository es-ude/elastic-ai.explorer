from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
import shutil
from socket import error as socket_error
import sys
import time

from fabric import Connection
import serial
from paramiko.ssh_exception import AuthenticationException

from elasticai.explorer.config import DeploymentConfig


class SSHException(Exception):
    pass


class Host(ABC):
    @abstractmethod
    def __init__(self, deploy_cfg: DeploymentConfig):
        pass

    @abstractmethod
    def put_file(self, local_path: str, remote_path: str | None) -> str:
        pass

    @abstractmethod
    def run_command(self, command: str) -> str:
        pass


class RPiHost(Host):
    def __init__(self, deploy_cfg: DeploymentConfig):
        self.host_name = deploy_cfg.target_name
        self.user = deploy_cfg.target_user
        self.logger = logging.getLogger(
            "explorer.platforms.deployment.device_communication.Host"
        )

    def _get_connection(self):
        return Connection(host=self.host_name, user=self.user)

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
                "{2}".format(command, self.host_name, str(result.stderr))
            )
        return result.stdout

    def put_file(self, local_path: str, remote_path: str | None) -> str:
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
                host=self.host_name, user=self.user, exc=exc
            )
        )


class PicoHost(Host):
    def __init__(self, deploy_cfg: DeploymentConfig):
        self.BAUD_RATE = 115200
        self.host_name = deploy_cfg.target_name
        self.logger = logging.getLogger(
            "explorer.platforms.deployment.device_communication.PicoHost"
        )
        self.serial_port = deploy_cfg.serial_port
        self.timeout_s = 40

    def _get_measurement(self) -> str:
        self._wait_for_pico(self.serial_port)
        line = ""
        try:
            with serial.Serial(self.serial_port, self.BAUD_RATE, timeout=1) as ser:
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

    def put_file(self, local_path: str, remote_path: str | None) -> str:
        time_passed = 0
        sleep_intervall = 0.5
        self.logger.info("Wait for pico to deploy...")
        while not os.path.isdir(self.host_name):
            time.sleep(sleep_intervall)
            time_passed = time_passed + sleep_intervall
            if time_passed > self.timeout_s:
                time.sleep(4)
                self.logger.error("Timeout on Pico-Communication")
                self.logger.info("Manual Reboot neccessary")

        shutil.copyfile(
            local_path,
            self.host_name + "/" + Path(local_path).name,
        )
        return self._get_measurement()

    def _wait_for_pico(self, port):
        self.logger.info("Wait for pico answer on Port " + port + "...")
        time_passed = 0
        sleep_intervall = 0.5
        while not os.path.exists(port):
            time.sleep(sleep_intervall)
            time_passed = time_passed + sleep_intervall
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

    def run_command(self, command: str) -> str:
        return ""
