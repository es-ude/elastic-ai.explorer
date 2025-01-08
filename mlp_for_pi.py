from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import KnowledgeRepository
from elasticai.explorer.platforms.deployment.manager import PIHWManager
from elasticai.explorer.train_model import train, test
from settings import ROOT_DIR


def find_and_generate_for_pi():
    knowledge_repository = KnowledgeRepository()
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


def take_measurements():
    knowledge_repository = KnowledgeRepository()
    explorer = Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    measurements = []
    model_path = str(ROOT_DIR) + "/models/ts_models/model_0.pt"
    for i in range(20):
        measurements.append(explorer.run_measurement(model_path))
    print(measurements)


def prepare_and_take_measurements():
    hw_manager = PIHWManager()
    hw_manager.compile_code()


if __name__ == "__main__":
    prepare_and_take_measurements()
    # take_measurements()
