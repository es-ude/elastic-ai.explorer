from optuna.trial import FixedTrial
import json
import yaml
from pathlib import Path
from torch import nn

from elasticai.explorer.hw_nas.hw_nas import sample_and_create_model


def reconstruct_model_from_json(file2json: Path, file2search: Path) -> nn.Sequential:
    """Reconstructing the model from explored model search defined in json file
    :param file2json:       Path to json file with model parameters
    :param file2search:     Path to yaml file with search space
    :return:                Sequantial with the PyTorch model
    """
    if not (file2json.is_file() and file2json.suffix == ".json"):
        raise FileNotFoundError(f"json file is not available at: {file2json.as_posix()}")
    with open(file2json.as_posix(), "r", encoding="utf-8") as f0:
        data = json.load(f0)[0]

    if not file2search.is_file() or not file2search.suffix in [".yaml", ".yml"]:
        raise FileNotFoundError(f"Search space file is not available at: {file2search.as_posix()}")
    with open(file2search.as_posix(), "r", encoding="utf-8") as f1:
        search = yaml.safe_load(f1)

    fixed_trial = FixedTrial(
        params=data,
        number=0
    )
    return sample_and_create_model(
        trial=fixed_trial,
        search_space=search
    )
