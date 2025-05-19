from typing import Optional, Callable

import nni
import yaml
from nni.mutable import  Mutable

from nni.nas.nn.pytorch import ModelSpace, LayerChoice, MutableLinear, Repeat,

from torch import nn
from torch.nn import Linear, ModuleList


# op_candidates = {
#     'linear': lambda in_feat, out_feat, activation: LinearActivation(in_feat, out_feat, activation),
#     'conv2d': lambda num_features: Conv1x1BNReLU(num_features, num_features),
#     'maxpool3x3': lambda num_features: nn.MaxPool2d(3, 1, 1)
# }


class LinearActivation(nn.Sequential):

    def __init__(self, in_feat, out_feat, activation: Callable[..., nn.Module]):
        super().__init__(
            MutableLinear(in_feat, out_feat),
            activation()
        )



activation_candidates = {
    "relu" : nn.ReLU(),
    "sigmoid" : nn.Sigmoid(),
    "identity": nn.Identity(),
    "tanh": nn.Tanh()
}

class SearchSpace(ModelSpace):

    def buildBlock(self, block: dict, input_width:Optional[int| Mutable], output_width: Optional[int]=None):
        block_name=block["Block"]
        op_candidates: dict=block["linear"]["activation"]
        activation_mappings=[activation_candidates[key] for key in op_candidates]

        depth: list = (list(block["depth_min_max"]))
        layers=[]

        activation= LayerChoice(activation_mappings, label=f"activation_block_{block_name}")

        h_l_widths = [nni.choice(f"layer_width_{i}_block_{block_name}", block["linear"]["width"]) for i in range(depth[1])]
        layers.append( nn.Sequential(MutableLinear(input_width, h_l_widths[0]),activation))

        #brauche builder der relu auch drin hat
        depth[1]= (depth[1]-1)

        repetitions: tuple[int, int]= tuple[int, int](depth)

        layers.append(Repeat(lambda index: nn.Sequential(MutableLinear(h_l_widths[index], h_l_widths[index+1]),activation), repetitions, label=f"depth_block_{block_name}"))

        layer_new=[]
        for layer in layers:
            layer_new = layer_new + [module for module in layer.modules() if not isinstance(module, nn.Sequential) and not isinstance(module, ModuleList) and  not isinstance(module, Repeat)]

        index = -1
        while not isinstance(layer_new[index], Linear):
            index -= 1
        layer = layer_new[index]
        last_layer = layer.out_features
        if output_width is not None:

            layers.append(
                nn.Sequential(MutableLinear(last_layer, output_width),
                           activation))
            return nn.Sequential(*layers), None


        return nn.Sequential(*layers), last_layer


    def __init__(self, parameters: dict):
        super().__init__()
        input_width=parameters["input"]
        output_width=parameters["output"]
        blocks: list[dict]= parameters["Blocks"]
        block_sp=[]

        block, last_out=self.buildBlock(blocks[0], input_width, None)
        block_sp.append(block)
        block, last_out=self.buildBlock(blocks[1], last_out, output_width)
        block_sp.append(block)

        self.block_sp=nn.Sequential(*block_sp)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x= self.block_sp(x)
        return x



def yml_to_dict(file):
    with open(file) as stream:
        try:
            search_space=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return search_space




if __name__=="__main__":
    search_space=yml_to_dict("search_space.yml")
    search_space=SearchSpace(search_space)
    print(search_space)


