import json
import os
from pathlib import Path
import tomllib
from typing import Any
import pandas as pd


def save_list_to_json(list: list, path_to_dir: Path, filename: str):
    os.makedirs(path_to_dir, exist_ok=True)
    with open(path_to_dir / filename, "w+") as outfile:
        json.dump(list, outfile)


def save_to_toml(toml_str: str, path_to_dir: Path, filename: str):
    os.makedirs(path_to_dir, exist_ok=True)
    with open(path_to_dir / filename, "w") as outfile:
        outfile.write(toml_str + "\n")


def load_toml(path_to_toml: Path) -> dict:
    with open(path_to_toml, "rb") as f:
            return tomllib.load(f)

def load_json(path_to_json: Path | str) -> Any:
    with open(path_to_json, "r") as f:
        return json.load(f)


def read_csv(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path)
