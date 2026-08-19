from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MAIN_EXPERIMENT_DIR = ROOT_DIR / "experiments"
CROSSCOMPILE_ENV_DIR = ROOT_DIR / "crosscompile-environments"
RBPI_CROSSCOMPILE_DIR = CROSSCOMPILE_ENV_DIR / "rbpi"
PICO_CROSSCOMPILE_DIR = CROSSCOMPILE_ENV_DIR / "pico"

