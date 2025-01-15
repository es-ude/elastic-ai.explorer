from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import KnowledgeRepository, HWPlatform
from elasticai.explorer.platforms.deployment.manager import PIHWManager, ConnectionData
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.train_model import train, test
from settings import ROOT_DIR


def setup_knowledge_repository():
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


def find_and_generate_for_pi(knowledge_repository, device_connection, max_search_trials):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    explorer.generate_search_space()
    top_models = explorer.search(max_search_trials)

    explorer.hw_setup_on_target(device_connection)
    measurements = []
    for i, model in enumerate(top_models):
        train(model, 3)
        test(model)
        model_path = str(ROOT_DIR) + "/models/ts_models/model_" + str(i) + ".pt"
        data_path = str(ROOT_DIR) + "/data"
        explorer.generate_for_hw_platform(model, model_path)
        measurements.append(
            explorer.run_measurement(device_connection, model_path)
        )

    measurements.append(
            explorer.verify_accuracy(device_connection, model_path, data_path)
        )
    print(measurements)


def take_measurements(knowledge_repository, connection_data):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    measurements = []
    model_path = str(ROOT_DIR) + "/models/ts_models/model_0.pt"
    explorer.hw_setup_on_target(device_connection)
    for i in range(20):
        measurements.append(explorer.run_measurement(connection_data, model_path))
    print(measurements)

def verify_model(knowledge_repository, connection_data):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    explorer.hw_setup_on_target(device_connection)
    measurements = []
    model_path = str(ROOT_DIR) + "/models/ts_models/model_0.pt"
    data_path = str(ROOT_DIR) + "/data"
    measurements.append(
            explorer.verify_accuracy(device_connection, model_path , data_path)
        )
    print(measurements)


def prepare_pi():
    hw_manager = PIHWManager()
    hw_manager.compile_code()


if __name__ == "__main__":
    ##Params
    host = "transfair.local"
    user = "robin"
    max_search_trials = 1



    knowledge_repo = setup_knowledge_repository()
    device_connection = ConnectionData(host, user)
    find_and_generate_for_pi(knowledge_repo, device_connection, max_search_trials)
    #verify_model(knowledge_repo, device_connection)
    #take_measurements(knowledge_repo, device_connection)
    #find_and_generate_for_pi(knowledge_repo, None)
