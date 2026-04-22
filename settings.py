import os
from pathlib import Path

# If the default (parent of this file's path) does not work, set the environment variable PROJECT_ROOT.
ROOT_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
DOCKER_CONTEXT_DIR = ROOT_DIR / "docker"
