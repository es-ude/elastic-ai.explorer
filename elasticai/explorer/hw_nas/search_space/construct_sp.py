from typing import Optional, Callable

import nni
import torch
import yaml
from nni.mutable import Categorical, Mutable
from nni.nas.hub.pytorch.nasnet import ReLUConvBN
from nni.nas.hub.pytorch.proxylessnas import ConvBNReLU
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, MutableLinear, Repeat, ParametrizedModule
from numpy.ma.core import argmax
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


#num_heads = [nni.choice(f"num_head_{i}", list(search_num_heads)) for i in range(max(search_depth))]



# def builder(index):
#     return nn.Sequential(MutableLinear(h_l_widths[index], h_l_widths[index+1]),nn.ReLU()), repetitions, label=f"depth")
#
#def map_strings_to_ops(ops:list[str])->list[Callable[nn.Module]]


activation_candidates = {
    "relu" : nn.ReLU(),
    "sigmoid" : nn.Sigmoid(),
    "identity": nn.Identity(),
    "tanh": nn.Tanh()
}

class SearchSpace(ModelSpace):

    def buildBlock(self, block: dict, input_width: int, output_width: int):




        op_candidates: dict=block["linear"]["activation"]
        print(op_candidates)
        activation_mappings=[activation_candidates[key] for key in op_candidates]
        print(activation_mappings)

        depth: list = (list(block["depth_min_max"]))
        layers=[]

        activation= LayerChoice(activation_mappings, label="activation")

        h_l_widths = [nni.choice(f"layer_width_{i}", block["linear"]["width"]) for i in range(depth[1])]
        layers.append( nn.Sequential(MutableLinear(input_width, h_l_widths[0]),activation))

        #brauche builder der relu auch drin hat
        depth[1]= (depth[1]-1)
        repetitions: tuple[int, int]= tuple[int, int](depth)

        layers.append(Repeat(lambda index: nn.Sequential(MutableLinear(h_l_widths[index], h_l_widths[index+1]),activation), repetitions, label=f"depth"))



        layer_new=[]
        for layer in layers:
            layer_new = layer_new + [module for module in layer.modules() if not isinstance(module, nn.Sequential) and not isinstance(module, ModuleList) and  not isinstance(module, Repeat)]
        print(layer_new)
        index=-1
        while not isinstance(layer_new[index], Linear):
            print(index)
            index-=1


        layer= layer_new[index]
        last_layer=layer.out_features
        layers.append(
            nn.Sequential(MutableLinear(last_layer, output_width),
                           nn.Sigmoid()))


        self.layers= nn.Sequential(*layers)


    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x= self.layers(x)
        return x

    def build_op_candidate(self, op_candidate: str, op_candidate_params: dict):
        if op_candidate == "linear" :
            return Linear()


    def __init__(self, parameters: dict):
        super().__init__()
        input_width=parameters["input"]
        output_width=parameters["output"]
        blocks: list[dict]= parameters["Blocks"]
        block_sp=[]
        for block in blocks:

            block_sp.append(self.buildBlock(block, input_width, output_width))
            # h1 = nni.choice("layer_1", [4, 32, 64, 512, 4096])
            # h2 = nni.choice("layer_2", [4, 32, 64, 4096])
            # h3 = nni.choice("dropout", [0.2, 0.25, 0.9])
            # self.fc1 = MutableLinear(28 * 28, h1)
            # self.fc2 = MutableLinear(h1, h2)
            # self.fc3 = MutableLinear(h2, 10)
            # self.dropout = MutableDropout(h3)




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
    sigmoid=nn.Sigmoid()
    tensor=torch.rand([32, 10])

    sig=sigmoid(tensor)
    print(sig.argmax(dim=1, keepdim=True))

