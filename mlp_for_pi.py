from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import KnowledgeRepository, HWPlatform
from elasticai.explorer.platforms.deployment.manager import PIHWManager, ConnectionData
from elasticai.explorer.platforms.generator.generator import PIGenerator
from elasticai.explorer.train_model import train, test
from settings import ROOT_DIR


def setup_knowledge_repository():
    knowledge_repository = KnowledgeRepository()
    device = ConnectionData("transpi5.local", "ies")
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "rpi5",
            "Raspberry PI 5 with A76 processor and 8GB RAM",
            PIGenerator,
            PIHWManager,
            device,
        )
    )
    return knowledge_repository


def find_and_generate_for_pi(knowledge_repository):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    explorer.generate_search_space()
    top_models = explorer.search()
    measurements = []
    for i, top_model in enumerate(top_models):
        train(top_model)
        test(top_model)
        model_path = str(ROOT_DIR) + "/models/ts_models/model_" + str(i)
        ts_model = explorer.generate_for_hw_platform(top_model, model_path)
        measurements.append(explorer.run_measurement(model_path + ".pt"))

    print(measurements)


def take_measurements(knowledge_repository):
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    measurements = []
    model_path = str(ROOT_DIR) + "/models/ts_models/model_0.pt"
    for i in range(20):
        measurements.append(explorer.run_measurement(model_path))
    print(measurements)


def prepare_and_take_measurements():
    hw_manager = PIHWManager(ConnectionData("transpi5.local", "ies"))
    hw_manager.compile_code()


if __name__ == "__main__":
    knowledge_repo = setup_knowledge_repository()
    find_and_generate_for_pi(knowledge_repo)
