# Generator
The Generator package is defining base classes and default implementations for the hardware-specific deployment of AI-models and corresponding measurement scripts. \
Given general and abstract base classes the Generator classes give flexibility in the implementation. The goal is to give a structure to the process of measuring empiric data on device.
However, they should follow this basic idea.

**Basic Idea**: 
- A ModelBuilder specifies how the Torch-Model has to be build to be feasible for the target hardware. 
- A ModelTranslator translates the Torch-Model to a Hardware Specific Model instance.
- A Compiler compiles the Hardware Specific Model (together with additional sources) to a deployable executable. 
- A Host gives basic functionalities to communicate with the target hardware.
- A HWManager takes the Compiler and Host to automate the deployment and also parses the received results. 
- The Generator dataclass simply specifies all the Generator Classes used for a specific target.

## Implementing a ModelBuilder 
If Generator has no specific requirements on the torch model, you don't need to implement your own ModelBuilder. The DefaultModelBuilder is used by default, allowing all the operations that are specified in the HW-NAS part, for more on this see [here](../hw_nas/search_space/README.md).

A ModelBuilder itself only require the implementation of a single function `build_from_trial(trial, searchspace)` which return the sampled model and optionally a quantization scheme. For a basic implementation, we recommend looking at the [DefaultModelBuilder](../hw_nas/search_space/build_model.py).
 
However, the ModelBuilder has a special role in the elastic-ai.Explorer, since it acts as the connecting layer between the HW-NAS and the Generator, specifying the capabilities of the Generator regarding supported types of model architectures.
Therefore, it inherits the abstract class Reflective, which defines this functionality. 

### Reflective
Reflective defines the capabilities of your Generator by defining mappings of identifier to allowed components. Returning empty dictionaries is seen as: "All default types are allowed". 
For the activation mappings, adapter mappings and layer mappings the key corresponds to the identifier you would give in the search space. The value is the instance or type of the class. The underlying registries of supported components are automatically updated on initialization. Reflective is initialized with the parameter `replace_registries`, if `true` this overwrites the default registries, else it only updates the registries with the additional capabilities given in Reflective. 

Example:
```python 
    def get_activation_mappings(self) -> dict[str, nn.Module]:
        return {"relu": nn.Relu(), "sigmoid": nn.Sigmoid()}

    def get_adapter_mappings(self) -> dict[tuple[str | None, str | None], None | type]:
        return {("linear": "conv2d"): LinearConv2DAdapter}

    def get_layer_mappings(self) -> dict[str, type[LayerBuilder]]:
        return {"linear": LinearLayerBuilder, "conv2d": Conv2DLayerBuilder}
```

Additionally, the supported quantization are given as constraints on specific parameters. Either sets or boolean functions can be given to specify the constraints. Returning an empty dictionary again means: "No constraints on quantization". 

Example:
```python 
    def get_supported_quantization(self) -> dict[str, Any]:
        return {"dtype": {"int8", "float16"}, "total_bits": lambda x: x <= 16}
```

A main functionality Reflective gives us is that we can call `Reflective.validate_model(self, model, quantization_scheme)` to check if a model can be used given the constraints. This raises specific exceptions, if types inside the model are not supported by the generator or the given quantization scheme is not supported. These exceptions are caught during traversal of the search space, leading to pruned samples. 



## Implementing a ModelTranslator
A ModelTranslate has to only impletranslate


## Implementing a Compiler

## Implementing a Host

## Implementing a HWManager