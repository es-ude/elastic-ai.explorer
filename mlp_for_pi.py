from elasticai.explorer.explorer import Explorer
from elasticai.explorer.knowledge_repository import KnowledgeRepository
from elasticai.explorer.train_model import train, test


def find_and_generate_for_pi():
    knowledge_repository= KnowledgeRepository()
    explorer= Explorer(knowledge_repository)
    explorer.choose_target_hw("rpi5")
    explorer.generate_search_space()
    top_models=explorer.search()
    for i, top_model in enumerate(top_models):
        train(top_model)
        test(top_model)
        ts_model=explorer.generate_for_hw_platform(top_model, path="models/ts_models/model_" +str(i) )
        test(ts_model)



if __name__ == '__main__':
    find_and_generate_for_pi()

