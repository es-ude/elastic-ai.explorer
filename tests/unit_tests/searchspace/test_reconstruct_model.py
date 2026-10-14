from elasticai.explorer.hw_nas.reconstruct_model import reconstruct_model_from_json
from elasticai.explorer import get_path_to_project

from torch.nn import Sequential


def test_reconstruct_model():
    path2files = (get_path_to_project() / "tests" / "unit_tests" / "searchspace" / "files").resolve().absolute()
    model = reconstruct_model_from_json(
        file2json=path2files / "models.json",
        file2search=path2files / "mnist_search_space.yaml",
    )
    assert isinstance(model, Sequential)
