import shutil
import sys
import tarfile
import tempfile
import zipfile
from enum import Enum
from pathlib import Path

import owncloud

from elasticai.explorer.platforms.deployment.device_communication import Host, RPiHost, SSHParams
from settings import DOCKER_CONTEXT_DIR

LIBTORCH_SCIEBO_URL = "https://uni-duisburg-essen.sciebo.de/s/9aiYf5Y2NABtdQb"

LIBTORCH_ARCHIVES: dict[str, str] = {
    "rpi4": "libtorch-v2.5.1-rpi4-bookworm.tar.gz",
    "rpi5": "libtorch-v2.6.0-rpi5-bookworm.zip",
}

class PiModel(Enum):
    RPi4 = "rpi4"
    RPi5 = "rpi5"


def _open_sciebo_client() -> owncloud.Client:
    return owncloud.Client.from_public_link(LIBTORCH_SCIEBO_URL)


def _raw_extract(archive_path: Path, dest: Path) -> None:
    if ".tar" in archive_path.name:
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest)
    elif archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    else:
        raise ValueError(f"Unsupported format: {archive_path.name}")


def _extract_to(archive_path: Path, target_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _raw_extract(archive_path, tmp_path)

        src = tmp_path / "libtorch"

        if not src.exists() or not src.is_dir():
            raise FileNotFoundError(f"No 'libtorch' directory in: {archive_path.name}'")

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target_dir))


def _download_archive(archive_name: str) -> Path:
    tmp_dir = tempfile.mkdtemp()
    archive_path = Path(tmp_dir) / archive_name
    client = _open_sciebo_client()
    client.get_file(archive_name, str(archive_path))
    client.logout()
    return archive_path


def _pi_install_command(archive_name: str) -> str:
    if archive_name.endswith(".zip"):
        extract = f'python3 -m zipfile -e ~/{archive_name} "$TMP"'
    else:
        extract = f'tar xzf ~/{archive_name} -C "$TMP"'

    return (
        "set -e && "
        'TMP="$(mktemp -d)" && '
        f"{extract} && "
        "sudo rm -rf /code/libtorch && "
        "sudo mkdir -p /code && "
        'sudo mv "$TMP/libtorch" /code/libtorch && '
        f'rm -rf "$TMP" ~/{archive_name}'
    )


def download_libtorch(pi_model: PiModel, target_dir: Path) -> None:
    archive_name = LIBTORCH_ARCHIVES[pi_model.value]
    archive_path = _download_archive(archive_name)
    try:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        _extract_to(archive_path, target_dir)
    finally:
        shutil.rmtree(archive_path.parent, ignore_errors=True)


def setup_docker_libtorch(pi_model: PiModel) -> Path:
    target = DOCKER_CONTEXT_DIR / "code" / "libtorch"
    download_libtorch(pi_model, target)
    return target


def install_libtorch_on_pi(pi_host: Host, pi_model: PiModel) -> None:
    archive_name = LIBTORCH_ARCHIVES[pi_model.value]
    archive_path = _download_archive(archive_name)
    try:
        pi_host.put_file(archive_path, ".")
    finally:
        shutil.rmtree(archive_path.parent, ignore_errors=True)
    pi_host.run_command(_pi_install_command(archive_name))


def _parse_args(argv: list[str]) -> dict:
    _args: dict = {"model": None, "docker": False, "pi_host": None, "pi_user": "pi"}
    i = 0
    while i < len(argv):
        if argv[i] in ("rpi4", "rpi5"):
            _args["model"] = argv[i]
        elif argv[i] == "--docker":
            _args["docker"] = True
        elif argv[i] == "--pi-host":
            i += 1
            _args["pi_host"] = argv[i]
        elif argv[i] == "--pi-user":
            i += 1
            _args["pi_user"] = argv[i]
        else:
            raise ValueError(f"Unknown argument: {argv[i]}")
        i += 1
    return _args


if __name__ == "__main__":
    """
        CLI Usage: 
        $ python -m elasticai.explorer.platforms.deployment.libtorch_installer {rpi4|rpi5} [--docker] [--pi-host HOST] [--pi-user USER]
        --docker          Extract into Docker build context
        --pi-host HOST    Pi hostname for SSH installation
        --pi-user USER    Pi SSH username (default: pi)
    """

    args = _parse_args(sys.argv[1:])

    if args["model"] is None:
        raise ValueError("Pi model (rpi4 or rpi5) is required")

    if not args["docker"] and not args["pi_host"]:
        raise ValueError("Specify at least one of --docker or --pi-host")

    pi_model = PiModel(args["model"])

    if args["docker"]:
        path = setup_docker_libtorch(pi_model)

    if args["pi_host"]:
        host = RPiHost(SSHParams(hostname=args["pi_host"], username=args["pi_user"]))
        install_libtorch_on_pi(host, pi_model)
