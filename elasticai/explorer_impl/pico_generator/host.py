from elasticai.explorer.generator.deployment.device_communication import (
    SerialHost,
    SerialParams,
)


import serial


import logging
import os
import shutil
import time
from pathlib import Path


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
            if time_passed > 32:
                raise TimeoutError(f"{self.host_name} not found")
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
