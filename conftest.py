def pytest_sessionstart(session):
    from elasticai.explorer import get_path_to_project
    from shutil import rmtree
    import logging

    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("explorer").setLevel(logging.WARNING)
    rmtree(get_path_to_project("experiments"), ignore_errors=True)


def pytest_sessionfinish(session, exitstatus):
    pass
