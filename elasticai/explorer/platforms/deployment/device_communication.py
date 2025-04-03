import logging
from socket import error as socket_error

from fabric import Connection
from paramiko.ssh_exception import AuthenticationException

from elasticai.explorer.config import ConnectionConfig


class SSHException(Exception):
    pass


class Host:
    def __init__(self,
                 connection_config: ConnectionConfig):
        self.host_name = connection_config.target_name
        self.user = connection_config.target_user
        self.logger = logging.getLogger("explorer.platforms.deployment.device_communication.Host")

    def _get_connection(self):
        return Connection(host=self.host_name, user=self.user)

    def run_command(self, command: str) -> str:
        try:
            with self._get_connection() as conn:
                self.logger.info("Install program on target. Hostname: %s - User: %s", conn.host,
                                 conn.user)
                result = conn.run(command, warn=True, hide=True)
        except (socket_error, AuthenticationException) as exc:
            self._raise_authentication_err(exc)

        if result.failed:
            raise SSHException(
                'The command `{0}` on host {1} failed with the error: '
                '{2}'.format(command, self.host_name, str(result.stderr)))
        return result.stdout

    def put_file(self, local_path: str, remote_path: str):
        try:
            with self._get_connection() as conn:
                conn.put(local_path, remote_path)
        except (socket_error, AuthenticationException) as exc:
            self._raise_authentication_err(exc)

    def _raise_authentication_err(self, exc):
        raise SSHException(
            "SSH: could not connect to {host} "
            "(username: {user}): {exc}".format(
                host=self.host_name, user=self.user, exc=exc))
