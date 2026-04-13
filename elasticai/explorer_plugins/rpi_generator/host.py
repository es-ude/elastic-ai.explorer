from elasticai.explorer.generator.deployment.device_communication import (
    SSHException,
    SSHHost,
    SSHParams,
)
from paramiko.ssh_exception import AuthenticationException


import logging
from pathlib import Path
from socket import error as socket_error


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
