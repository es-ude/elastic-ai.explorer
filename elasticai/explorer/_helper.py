from pathlib import Path


def get_path_to_project(new_folder: str = "") -> Path:
    """Function for getting the path to find the project folder structure in application.
    :return:            Absolute path to start the project structure
    """
    max_levels: int = 5

    cwd = Path(".").absolute()
    current = cwd

    for _ in range(max_levels):
        if (current / "pyproject.toml").exists():
            break
        current = current.parent
    return (current / new_folder).resolve().absolute()


def get_path_to_docker() -> Path:
    return get_path_to_project("docker")


def get_path_to_experiments() -> Path:
    return get_path_to_project("experiments")
