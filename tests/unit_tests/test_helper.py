from explorer._helper import get_path_to_project


def test_get_path_to_project_wo_ref():
    ref = ["elastic-ai", "explorer"]
    chck = get_path_to_project().as_posix()
    rslt = [True for key in ref if key in chck]
    assert sum(rslt) == 2


def test_get_path_to_project_with_ref():
    chck = get_path_to_project(new_folder="test").as_posix()
    rslt = (get_path_to_project() / "test").as_posix()
    assert rslt == chck