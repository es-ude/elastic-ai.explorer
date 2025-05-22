from typing import Optional, Callable

import nni
import yaml
from nni.mutable import  Mutable

from nni.nas.nn.pytorch import ModelSpace, LayerChoice, MutableLinear, Repeat

from torch import nn
from torch.nn import Linear, ModuleList



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
}




class SearchSpace(ModelSpace):
    def get_last_layer_width(self, layers):
        layer_new = []
        for layer in layers:
            layer_new = layer_new + [module for module in layer.modules() if
                                     not isinstance(module, nn.Sequential) and not isinstance(module,
                                                                                              ModuleList) and not isinstance(
                                         module, Repeat)]

        index = -1
        while not isinstance(layer_new[index], Linear):
            index -= 1
        layer = layer_new[index]
        last_layer = layer.out_features
        return last_layer



    def buildBlock(self, block: dict, input_width:Optional[int| Mutable], output_width: Optional[int]=None):
        block_name=block["Block"]
        op_candidates: dict=block["linear"]["activation"]
        activation_mappings=[activation_candidates[key] for key in op_candidates]

        depth= block["depth"]
        if isinstance(depth, int):
            max_depth=depth
            repetitions=depth
        else:
            max_depth=depth[1]

            repetitions= tuple[int, int](depth)

        layers=[]

        activation= LayerChoice(activation_mappings, label=f"activation_block_{block_name}")
        self.h_l_widths = [input_width]
        self.h_l_widths=self.h_l_widths+ [nni.choice(f"layer_width_{i}_block_{block_name}", block["linear"]["width"]) for i in range(max_depth)]

        layers.append(Repeat(lambda index: nn.Sequential(MutableLinear(self.h_l_widths[index], self.h_l_widths[index+1]),activation), repetitions, label=f"depth_block_{block_name}"))

        last_layer=self.get_last_layer_width(layers)
        if output_width is not None:

            layers.append(
                nn.Sequential(MutableLinear(last_layer, output_width),
                           activation))

            return nn.Sequential(*layers), None


        return nn.Sequential(*layers), last_layer


    def __init__(self, parameters: dict):
        super().__init__()
        blocks: list[dict]= parameters["Blocks"]
        block_sp=[]

        last_out=None
        for block in blocks:
            if last_out is None:
                input_width=parameters["input"]
            #    block, last_out=self.buildBlock(block, input_width, None)
            else:
                input_width=last_out
            if blocks[-1]["Block"] == block["Block"]:
                output_width=parameters["output"]
            else:
                output_width=None

            block, last_out = self.buildBlock(block, input_width, output_width)
            block_sp.append(block)

      #  block_sp.append(block)

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


