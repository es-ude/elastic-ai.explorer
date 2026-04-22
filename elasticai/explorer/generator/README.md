# Generator
The Generator package is defining abstract base classes and some default implementations for the hardware-specific deployment of AI-models and corresponding measurement scripts. \
Given general and abstract base classes the Generator classes give flexibility in the implementation. The goal is to give a structure to the process of measuring empiric data on device.
However, they should follow this basic idea.

**Basic Idea**: 
- A ModelBuilder specifies how the Torch-Model has to be build to be feasible for the target hardware. 
- A ModelTranslator translates the Torch-Model to a Hardware Specific Model instance.
- A Compiler compiles the Hardware Specific Model (together with additional sources) to a deployable executable. 
- A Host gives basic functionalities to communicate with the target hardware.
- A HWManager takes the Compiler and Host to automate the deployment and also parses the received results. 
- The Generator dataclass simply specifies all the Generator Classes used for a specific target.

Example implementation of all the components are located in the [pico-generator](../../explorer_impl/pico_generator/README.md) and [rpi-generator](../../explorer_impl/rpi_generator/README.md) packages. 

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
class ExampleModelBuilder(ModelBuilder)
    def get_activation_mappings(self) -> dict[str, nn.Module]:
        return {"relu": nn.Relu(), "sigmoid": nn.Sigmoid()}

    def get_adapter_mappings(self) -> dict[tuple[str | None, str | None], None | type]:
        return {("linear": "conv2d"): LinearConv2DAdapter}

    def get_layer_mappings(self) -> dict[str, type[LayerBuilder]]:
        return {"linear": LinearLayerBuilder, "conv2d": Conv2DLayerBuilder}
    ...
```

Additionally, the supported quantization are given as constraints on specific parameters. Either sets or boolean functions can be given to specify the constraints. Returning an empty dictionary again means: "No constraints on quantization". 

Example:
```python 
class ExampleModelBuilder(ModelBuilder)
    ...
    def get_supported_quantization(self) -> dict[str, Any]:
        return {"dtype": {"int8", "float16"}, "total_bits": lambda x: x <= 16}
    ...
```

A main functionality of Reflective is the member function `validate_model(self, model, quantization_scheme)` that checks if a model can be used given the defined constraints. This raises specific exceptions telling the user which components of the model or which type of quantization are not supported by the generator. These exceptions are then caught during traversal of the search space, leading to pruned samples. 

## Implementing a ModelTranslator
A ModelTranslate has requires the implementation of translate. This should create a model-file (e.g., .pt, .tflite) at the output-path or any instance that can be compiled by the Compiler, the specific implementation is completely flexible. Example implementations for this can be found in [Pico-Generator](../../explorer_impl/pico_generator/model_translator.py) and [RPI-Generator](../../explorer_impl/rpi_generator/model_translator.py). 

## Implementing a Compiler
A Compiler is initialized with the Dataclass CompilerParams to configure the Compiler instance. Additional parameter can be given in `kwargs`. 
Importantly this does not aim to be a universal Compiler Class but should allow for cross-compiling of simple measurement scripts of hardware metrics.  
A Compiler requires the definition of the following self-explanatory member functions:

```python 
    @abstractmethod
    def is_setup(self) -> bool:
        pass
    @abstractmethod
    def setup(self) -> None:
        pass
    @abstractmethod
    def compile_code(self, source: Path, output_dir: Path = Path("")) -> Path:
        pass
```
Example implementations (using docker cross-compilation) for this can be found in [Pico-Generator](../../explorer_impl/pico_generator/compiler.py) and [RPI-Generator](../../explorer_impl/rpi_generator/compiler.py). 


## Implementing a Host
The Host is split in the abstract base classes SSHHost and SerialHost, giving the most general capabilities to specify the communication between target hardware and PC with SSH and serial connection respectively. Example implementations of the classes can be found in [Pico-Generator](../../explorer_impl/pico_generator/host.py) and [RPI-Generator](../../explorer_impl/rpi_generator/host.py).

## Implementing a HWManager
The hardware manager defines all necessary functionality of repeated deployment and measurements. For that it handles a Host and Compiler instance. Optionally, specifics on test data and quantization are handled as well.  
The HWManager abstract base class gives some basic implementations which can or should be overwritten to customize behavior.  
Example implementations of can be found in [Pico-Generator](../../explorer_impl/pico_generator/hw_manager.py) and [RPI-Generator](../../explorer_impl/rpi_generator/hw_manager.py).

The general idea is to register the sources to the target metrics (e.g. Accuracy, Latency) you want to measure. The sources should be a path to source files (model, data, scripts) or for more customizability, a `MetricFunction = Callable[[Host, "HWManager"], dict[str, dict]]` which can be any Callable follows the only expected to return the results as a nested dictionary. This behavior is handled by the following member functions, that can be overridden:  

```python
    def prepare_measurement(self, source: Path | MetricFunction, metric: Metric) # By default this only registers the source to the metric.
    def _invoke_metric_source(self, metric: Metric, path_to_model: Path) # By default this only calls MetricFunctions.
```

Between or before measurements it may be necessary to prepare data and model (e.g. compilation). Which should be defined in the following member functions. 
```python
    def prepare_dataset(self, dataset_spec: DatasetSpecification, quantization_scheme: QuantizationScheme | None)
    def prepare_model(self, path_to_model: Path)
```

If everything is prepared the measurement takes place in `measure_metric(...)` given the specific model and metric. The results should be given as structured and readable dictionary.  
By default this only executes `_invoke_metric_source(...)`.
```python
def measure_metric(self, metric: Metric, path_to_model: Path) -> dict:
    return self._invoke_metric_source(metric, path_to_model)
```