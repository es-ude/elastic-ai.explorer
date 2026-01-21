
from elasticai.explorer.generator.model_builder.model_builder import CreatorModelBuilder
from elasticai.explorer.hw_nas.search_space.construct_search_space import SearchSpace
from optuna.trial import FixedTrial
class TestConstruct_SP:
    def test_build_creator_model(self, search_space_dict):
        search_space = SearchSpace(search_space_dict)

        sample = {
            "num_layers_b1": 1,
            "operation_b1": "linear",
            "layer_width_b1_l0": 2,
            "activation_func_b1_l0": "relu",
        }

        model_builder = CreatorModelBuilder()
        model, _ = model_builder.build_from_trial(FixedTrial(sample), search_space)
    
    