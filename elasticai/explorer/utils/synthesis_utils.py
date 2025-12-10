from fabric import Connection
from enum import StrEnum, auto, Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from tarfile import open as tar_open
from string import Template
from invoke.exceptions import UnexpectedExit
import os


class TargetPlatforms(StrEnum):
    env5_s50 = auto()
    env5_s15 = auto()


_fpga_model_for_platform = {
    TargetPlatforms.env5_s50: "xc7s50ftgb196-2",
    TargetPlatforms.env5_s15: "xc7s15ftgb196-2",
}

_SRCS_FILE_BASE_NAME = "synth_srcs"
_SRCS_FILE_NAME = f"{_SRCS_FILE_BASE_NAME}.tar.gz"
_TCL_SCRIPT_NAME = "autobuild.tcl"

_tcl_script_content_tpl = Template(
    """
# run.tcl V3 - exits on any error

# Function to handle errors and exit Vivado
proc exit_on_error {errorMsg} {
    puts "Error: $errorMsg"
    exit 1
}

# Top-level catch block
if {[catch {

    # STEP#1: Setup design sources and constraints
    create_project ${project_name} ${remote_working_dir} -part $part_number -force
    add_files -fileset sources_1 ${remote_working_dir}/${srcs_dir}
    add_files -fileset constrs_1 -norecurse ${remote_working_dir}/${srcs_dir}/constraints.xdc
    update_compile_order -fileset sources_1

    # STEP#2: Run synthesis
    launch_runs synth_1 -jobs ${num_jobs}
    wait_on_run synth_1

    # STEP#3: Run implementation
    set_property STEPS.WRITE_BITSTREAM.ARGS.BIN_FILE true [get_runs impl_1]
    launch_runs impl_1 -to_step write_bitstream -jobs ${num_jobs}
    wait_on_run impl_1

} errorMsg]} {
    exit_on_error $errorMsg
}

# Exit cleanly if everything succeeds
exit
"""
)


def _create_srcs_archive(tmp_dir: Path, src_dir: Path) -> Path:
    result = tmp_dir / _SRCS_FILE_NAME
    with tar_open(result, "w:gz") as tar_file:
        for src_file in src_dir.glob("**/*.vhd"):
            tar_file.add(src_file, str(src_file.relative_to(src_dir)))

        for src_file in src_dir.glob("**/*.xdc"):
            tar_file.add(src_file, "constraints.xdc")
        os.chdir(tmp_dir)
        tar_file.add(tmp_dir / _TCL_SCRIPT_NAME, _TCL_SCRIPT_NAME)
    return result


def _write_tcl_script(
    tmp_dir: Path,
    project_name: str,
    part_number: str,
    remote_working_dir: str,
    num_jobs: int,
):
    (tmp_dir / _TCL_SCRIPT_NAME).write_text(
        _tcl_script_content_tpl.safe_substitute(
            remote_working_dir=remote_working_dir,
            srcs_dir=_SRCS_FILE_BASE_NAME,
            project_name=project_name,
            num_jobs=num_jobs,
            part_number=part_number,
        )
    )


def remote_path_exists(connection, remote_path) -> bool:
    # There has to be a better way than this
    try:
        connection.run(f"ls {remote_path}")
        return True
    except UnexpectedExit:
        return False


def try_remove_recursively(connection, remote_path):
    if remote_path_exists(connection, remote_path):
        connection.run(f"rm -rf {remote_path}")


class Verbosity(Enum):
    ONLY_ERRORS = auto()
    OUT_AND_ERRORS = auto()
    ALL = auto()


class ConnectionWrapper:
    def __init__(self, wrapped: Connection, verbosity: Verbosity):
        self._wrapped = wrapped
        self._verbosity = verbosity

    def run(self, cmd):
        match self._verbosity:
            case Verbosity.ONLY_ERRORS:
                return self._wrapped.run(cmd, hide="out")
            case Verbosity.ALL:
                return self._wrapped.run(cmd, echo=True)
            case _:
                return self._wrapped.run(cmd)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def run_vhdl_synthesis(
    src_dir: Path,
    host: str,
    ssh_user: str,
    remote_working_dir: str,
    vivado_path: str = "/tools/Xilinx/Vivado/2023.1/bin/vivado",
    ssh_port: int = 22,
    target=TargetPlatforms.env5_s50,
    quiet: bool = False,
) -> Path:
    """Generate FPGA bitstreams remotely.

    The tool uploads files in SRC_DIR via ssh to a host,
    generates bitstream with vivado and downloads it
    together with reports.

    All options, except for --quiet, can be set from
    environment variables. Names for these variables
    are formed by prepending SYNTH_, e.g., --ssh-user
    becomes SYNTH_SSH_USER.
    """
    print(f"connecting to {host} as {ssh_user}")
    print(f"uploading {src_dir.absolute()}")
    src_dir = src_dir.absolute()
    target_file = src_dir.parent.absolute() / "vivado_run_results.tar.gz"
    project_name = "Explorer-Synthesis"
    connection = ConnectionWrapper(
        Connection(host=host, user=ssh_user, port=ssh_port),
        Verbosity.ONLY_ERRORS if quiet else Verbosity.ALL,
    )
    if target_file.exists():
        print(
            f"skipping {src_dir} because target file {target_file.absolute()} already exists"
        )
        return src_dir.parent.absolute() / "results/impl/env5_top_reconfig.bin"
    if "./" in remote_working_dir:
        raise ValueError("illegal remote working directory")
    with TemporaryDirectory(suffix="synth_server") as tmp_dir:
        print(f"preparing files in {tmp_dir}")
        tmp_dir = Path(tmp_dir)

        _write_tcl_script(
            tmp_dir,
            project_name=project_name,
            part_number=_fpga_model_for_platform[target],
            remote_working_dir=remote_working_dir,
            num_jobs=12,
        )
        print("archiving srcs")
        srcs_archive = _create_srcs_archive(tmp_dir, src_dir)
        print("uploading srcs to server")

        for name in [
            "*.log",
            "*.jou",
            "results",
            _SRCS_FILE_BASE_NAME,
        ]:
            try_remove_recursively(connection, f"{remote_working_dir}/{name}")
        connection.run(f"mkdir -p {remote_working_dir}/{_SRCS_FILE_BASE_NAME}")
        connection.put(
            srcs_archive,
            remote=f"{remote_working_dir}/{_SRCS_FILE_NAME}".removeprefix("~/"),
        )
    with connection.cd(remote_working_dir):
        print("unpacking srcs on server")
        connection.run(
            "tar -C {dir} -xzf {srcs}".format(
                dir=_SRCS_FILE_BASE_NAME, srcs=_SRCS_FILE_NAME
            )
        )
        print("starting vivado implementation run")
        connection.run(
            "{vivado} -mode tcl -source {srcs}/{tcl_script}".format(
                vivado=vivado_path,
                srcs=_SRCS_FILE_BASE_NAME,
                tcl_script=_TCL_SCRIPT_NAME,
            )
        )
        print("vivado done")
        for cmd in [
            "mkdir -p results/synth",
            "mkdir results/impl",
            f"cp {project_name}.runs/impl_1/*.bin results/impl/",
            f"cp {project_name}.runs/impl_1/*.rpt results/impl/",
            f"cp {project_name}.runs/synth_1/*.rpt results/synth/",
            "tar -czf results.tar.gz results",
        ]:
            connection.run(cmd)
    connection.get(
        local=str(src_dir.parent.absolute() / "vivado_run_results.tar.gz"),
        remote=f"{remote_working_dir}/results.tar.gz",
    )
    return src_dir.parent.absolute() / "results/impl/env5_top_reconfig.bin"
