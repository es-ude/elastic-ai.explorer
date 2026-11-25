import argparse
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

from elasticai.explorer import get_path_to_docker
from elasticai.explorer.platforms import PlatformID
from elasticai.explorer.platforms.deployment.device_communication import Host, RPiHost, SSHParams


@dataclass(frozen=True)
class LibTorchArchive:
    tag: str
    name: str
    base_url: str = "https://github.com/es-ude/elastic-ai.explorer/releases/download"

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.tag}/{self.name}"


class LibTorchInstaller:
    def __init__(self, archive: LibTorchArchive) -> None:
        self.archive = archive

    def install(self, install_locally: bool = True, pi_host: Host | None = None) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            filepath = self.download(Path(tmp))

            if install_locally:
                target_dir = get_path_to_docker() / "code"
                shutil.rmtree(target_dir / "libtorch", ignore_errors=True)
                target_dir.mkdir(parents=True, exist_ok=True)
                self.extract(filepath, target_dir)

            if pi_host:
                pi_host.put_file(filepath, ".")
                pi_host.run_command(self._remote_install_command)

    def download(self, destination: Path) -> Path:
        local_path = destination / self.archive.name
        urlretrieve(self.archive.url, local_path)
        return local_path

    @staticmethod
    def extract(origin: Path, destination: Path):
        with tarfile.open(origin) as tf:
            tf.extractall(destination, filter='data')

    @property
    def _remote_install_command(self) -> str:
        return (
            'set -e && '
            'TMP="$(mktemp -d)" && '
            f'tar xzf ~/{self.archive.name} -C "$TMP" && '
            'sudo rm -rf /code/libtorch && '
            'sudo mkdir -p /code && '
            f'sudo mv "$TMP/libtorch" /code/libtorch && '
            f'rm -rf "$TMP" ~/{self.archive.name}'
        )


LIBTORCH_ARCHIVES: dict[PlatformID, LibTorchArchive] = {
    PlatformID.RPI4: LibTorchArchive(
        tag="rpi4-libtorch",
        name="libtorch-v2.5.1-rpi4-bookworm.tar.gz",
    ),
    PlatformID.RPI5: LibTorchArchive(
        tag="rpi5-libtorch",
        name="libtorch-v2.6.0-rpi5-bookworm.tar.gz"
    )
}

def install_libtorch(
    platform: PlatformID,
    install_locally: bool = True,
    pi_host: Host | None = None,
) -> None:
    installer = LibTorchInstaller(LIBTORCH_ARCHIVES[platform])
    installer.install(install_locally=install_locally, pi_host=pi_host)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m elasticai.explorer.platforms.deployment.libtorch_installer",
        description="Install LibTorch for a Raspberry Pi 4 or 5.",
    )
    parser.add_argument(
        "hw_platform",
        choices=["rpi4", "rpi5"],
        help="Target hardware platform ('rpi4' or 'rpi5')",
    )
    parser.add_argument(
        "--docker", "-d",
        action="store_true",
        help="Extract into Docker build context",
    )
    parser.add_argument(
        "--pi-host", "-H",
        metavar="HOST",
        help="Pi hostname for SSH installation",
    )
    parser.add_argument(
        "--pi-user", "-u",
        metavar="USER",
        help="Pi SSH username",
    )

    _args = parser.parse_args()

    if not _args.docker and not _args.pi_host:
        parser.error("specify at least one of --docker or --pi-host")

    if _args.pi_host and not _args.pi_user:
        parser.error("--pi-host requires --pi-user")

    if _args.pi_user and not _args.pi_host:
        parser.error("--pi-user requires --pi-host")

    return _args


if __name__ == "__main__":
    args = _parse_args()

    install_libtorch(
        platform=PlatformID(args.hw_platform),
        install_locally=args.docker,
        pi_host=RPiHost(
            params=SSHParams(
                hostname=args.pi_host,
                username=args.pi_user
            )
        )
        if args.pi_host is not None else None
    )
