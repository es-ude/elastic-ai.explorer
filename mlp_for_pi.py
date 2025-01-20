from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import KnowledgeRepository, HWPlatform
from elasticai.explorer.platforms.deployment.manager import PIHWManager, ConnectionData
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.train_model import train, test
from settings import ROOT_DIR


def setup_knowledge_repository_pi5():
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi5",
            "Raspberry PI 5 with A76 processor and 8GB RAM",
            PIGenerator,
            PIHWManager,
        )
    )
    return knowledge_repository


def setup_knowledge_repository_pi4():
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi4",
            "Raspberry PI 4 with A72 processor and 4GB RAM",
            PIGenerator,
            PIHWManager,
        )
    )
    return knowledge_repository


def find_generate_measure_for_pi(
    knowledge_repository,
    device_connection,
    max_search_trials,
    pi_type="rpi5",
    path_to_libtorch="./code/libtorch",
):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw(pi_type)
    explorer.generate_search_space()
    top_models = explorer.search(max_search_trials)

    explorer.hw_setup_on_target(device_connection, path_to_libtorch)
    measurements = []
    for i, model in enumerate(top_models):
        train(model, 3)
        test(model)
        model_path = str(ROOT_DIR) + "/models/ts_models/model_" + str(i) + ".pt"
        data_path = str(ROOT_DIR) + "/data"
        explorer.generate_for_hw_platform(model, model_path)
        measurements.append(
            explorer.run_latency_measurement(device_connection, model_path)
        )

    print(
        "Accuracy: ", explorer.verify_accuracy(device_connection, model_path, data_path)
    )

    print("Latency in Microseconds: ", measurements)


def measure_latency(
    knowledge_repository,
    connection_data,
    path_to_libtorch="./code/libtorch",
    pi_type="rpi5",
):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw(pi_type)
    measurements = []
    model_path = str(ROOT_DIR) + "/models/ts_models/model_0.pt"
    explorer.hw_setup_on_target(device_connection, path_to_libtorch)
    for i in range(20):
        measurements.append(
            explorer.run_latency_measurement(connection_data, model_path)
        )
    print("Latencies: ", measurements)


def measure_accuracy(
    knowledge_repository,
    connection_data,
    path_to_libtorch="./code/libtorch",
    pi_type="rpi5",
):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw(pi_type)
    explorer.hw_setup_on_target(device_connection, path_to_libtorch)
    measurements = []
    model_path = str(ROOT_DIR) + "/models/ts_models/model_0.pt"
    data_path = str(ROOT_DIR) + "/data"

    print(
        "Accuracy: ", explorer.verify_accuracy(device_connection, model_path, data_path)
    )


def prepare_pi5():
    hw_manager = PIHWManager()
    hw_manager.compile_code()


if __name__ == "__main__":
    ##Params
    host = "transpi4.local"
    user = "transfair"
    max_search_trials = 1
    pi_type = "rpi4"
    torch_path = "./code/libtorch-v2.5.1-rpi4-bookworm/libtorch"

    # knowledge_repo = setup_knowledge_repository_pi5()
    knowledge_repo = setup_knowledge_repository_pi4()
    device_connection = ConnectionData(host, user)
    find_generate_measure_for_pi(
        knowledge_repo,
        device_connection,
        max_search_trials,
        pi_type="rpi4",
        path_to_libtorch="./code/libtorch-v2.5.1-rpi4-bookworm/libtorch",
    )
    # measure_accuracy(knowledge_repo, device_connection, torch_path, pi_type)
    # measure_latency(knowledge_repo, device_connection, torch_path, pi_type)
