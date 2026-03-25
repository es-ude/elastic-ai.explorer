# Search space YAML
 The search space is specified in a search space yaml which will be given into the hardware Nas component of the explorer 
 and translated to an optuna compatible search space. Then each trial the search parameters will be sampled which will
 be automatically translated to a pytorch model.
This section describes the Syntax of the yaml file to describe the search space.

### Search space Structure

Each search space definition needs to specify an input dimension, an output dimension and a sequence of blocks. Optionally, options for model-wide quantization schemes can be specified, if this isn't included it will default to Full-Precision (Float32). 
Optionally, a section describing the default search parameters for primitive operations and a section for defining custom composite operations can be added.
When the default_op_params doesn't include the default search parameters of an operation, the search parameters have to be defined in the block where they are used.
```yaml
input: [1, 1313]          # The dimensions of an Input Sample without batch_size
output: 2                 # The dimensions of the Output

quantization:             # It possible to specify quantization schemes that will applied model-wide. 
  quant_candidates: [...] # Which schemes are supported dependents on the Generator. 
    quant_candidate_1:    # These can also be parametrized individually.
      param_1: [...]

sequence:                 # The macro structure of the network consisting of multiple blocks (at least one)
  - block: "1"            # Each block needs to have a unique string identifier
      ...               
  - block: "2"
      ...
default_op_params:        # It is possible to specify the default search parameters for each primitive operation in this section of the yaml as a dictionary.
                          # If an operation is used somewhere and it is not specified here, it will have to be specified in the block directly.
    primitive_op_1:       # Supported primitive operations can be seen in Table 1. The naming must correspond.
      param_1: [...]      # Valid parameters differ for each primitive operation and can be seen further down
      param_2: [...]
    primitive_op_2:
      param_1: [..]

```


Each block has a unique name, a rule of how to repeat the block, the candidate operations and optionally, for each operation possible parameter configurations.
The type_repeat block can be omitted if the block should only be repeated once and does not reference a different block.
If not, there are different options for the type:

```yaml 
- block: "1"
    ...
- block: "2"
  type_repeat:
    type: "repeat_op" # Possible: "repeat_op", "repeat_params", "vary_all", "repeat_block", "mirror_block"
    depth: [ 2, 3] # Only for repeat_op, repeat_params, vary_all
    ref_block: "1" # Only for repeat_block, mirror_block
  op_candidates: ["operation_1, operation_2"]  #All the operations the nas should try in this block
  operation_1:
      param_1: [...]
      param_2: [...]
  operation_2:
      ...

```

| type          | Description                                                                                                      | Mandatory fields                                                                 |
|---------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| repeat_op     | Repeat the same operation for the chosen depth. Searched parameters for each layer may change.                   | depth must be set                                                                |
| repeat_params | Only search the operations and its parameters once per block. Reuse sampled params for every layer in the block. | depth must be set                                                                | 
| vary_all      | Vary operations and parameters for each layer in block                                                           | depth must be set                                                                |
| repeat_block  | Repeat an already defined block Input can vary depending on previous block                                       | reference_block must be set with unique block identifier of block to be repeated |
| mirror_block  | Mirror an already defined block. NOT IMPLEMENTED YET                                                             | reference_block must be set with unique block identifier of block to be mirrored |



| Supported Operations    | Mandatory fields                    | Optional Fields                                       |
|-------------------------|-------------------------------------|-------------------------------------------------------|
| linear                  | width(can be omitted in last layer) | activation                                            |
| identity                | none                                | none                                                  |
| conv2d                  | kernel_size, out_channels           | stride, padding                                       | 
| conv1d                  | kernel_size, out_channels           | stride, padding                                       |
| lstm                    | hidden_size                         | bidirectional, num_layers, dropout, batch_first, bias |
| activation              | op                                  | None                                                  |
| batch_norm              | None                                | None                                                  |
| avgpool                 | kernel_size                         | stride, padding                                       |
| maxpool                 | kernel_size                         | stride, padding                                       |                                                       |
| dropout                 | None                                | p                                                     |
| gaussian_dropout        | None                                | p                                                     |
| layer_norm              | None                                | None                                                  |
| repeat_vector           | times                               | None                                                  |
| time_distributed_linear | width                               | batch_first, activation                               |

| Supported Quantization  | Mandatory fields                    | Optional Fields                                       |
|-------------------------|-------------------------------------|-------------------------------------------------------|
| full_precision                 | None                       | None
| ptq_fully_quantized_int8       | None                       | None

### Composite
It's possible to combine Operations to a composite operation and reuse that in the rest of the search space. 
This is done by defining the composite section in the yaml file. The operation must be named and be a valid search space on its own.
It's important that the blocks are still named differently to the main searchspace.

```yaml
composites:
  conv-act-pool:
    sequence:
      - block: "conv_act_pool_1"
        op_candidates:  "conv1d"
        type_repeat:
          type: "repeat_params"
          depth:  2 

          padding:  "same" 
      - block: "conv_act_pool_2"
        op_candidates:  "activation" 
        activation:
          op:  "relu" 
      - block: "conv_act_pool_3"
        op_candidates: [ "maxpool", "avgpool"]
        maxpool:
          kernel_size: [ 2, 4 ]
        avgpool:
          kernel_size: [ 2, 4 ]

```
### How to specify search parameters
 The explorer accepts lists of parameters. In this case each of the specified values can be sampled:
```yaml 
    linear:
      width: [16, 64, 128] # In this case width can take on three different values.
```
It is also possible to specify parameter ranges for int and float values
For int if no step is specified the default will be step=1.
For float if no step is specified it will sample logarithmically.
```yaml 
    linear:
      width:
        start: 10
        end: 100
        step: 10
   ```
If the parameter should not be searched but should have a specific value the brackets should be omitted
```yaml 
    linear:
      width: 128
   ```
### Identity
It is possible to completely omit a layer or even a block by using the Identity operation.
In the case below either this block will be a 1d Convolution block or will not appear in the model at all.
```yaml 
    - block: "1"
      op_candidates: ["conv1d", "identity"]
      type_repeat:
          type: "repeat_op"
          depth: [  2, 6, 7 ]
   ```
## Complete Search Space Yaml Example
```yaml 
input: [1, 1313]
output: 2
sequence:
  - block: "1"
    op_candidates: ["conv1d", "linear"]
    type_repeat:
      type: "repeat_op"
      depth: [  2, 6, 7 ]
    linear:
      width: [ 4, 5,]
      activation: [ "sigmoid" , "relu"]
  - block:  "2"
    op_candidates: [ "linear"]
    type_repeat:
      type: "repeat_op"
      depth: [ 1, 2 , 3]
    linear:
      width: 128




default_op_params:
  conv1d:
    kernel_size: [ 6, 12 ]
    stride: [ 1,2 ]
    out_channels: [ 8, 10, 16 ]
    padding: [ "same"]
```
Example networks
```
Sequential(
  (0): Conv1d(1, 8, kernel_size=(6,), stride=(1,), padding=(True,))
  (1): Conv1d(8, 16, kernel_size=(12,), stride=(2,), padding=(True,))
  (2): Conv1d(16, 16, kernel_size=(6,), stride=(2,), padding=(True,))
  (3): Conv1d(16, 8, kernel_size=(6,), stride=(2,), padding=(True,))
  (4): Conv1d(8, 16, kernel_size=(12,), stride=(1,), padding=(True,))
  (5): Conv1d(16, 10, kernel_size=(12,), stride=(1,), padding=(True,))
  (6): ToLinearAdapter(
    (layer): Flatten(start_dim=1, end_dim=-1)
  )
  (7): Linear(in_features=1430, out_features=128, bias=True)
  (8): Linear(in_features=128, out_features=2, bias=True)
)

Sequential(
  (0): ToLinearAdapter(
    (layer): Flatten(start_dim=1, end_dim=-1)
  )
  (1): Sequential(
    (0): Linear(in_features=1313, out_features=4, bias=True)
    (1): Sigmoid()
  )
  (2): Sequential(
    (0): Linear(in_features=4, out_features=4, bias=True)
    (1): Sigmoid()
  )
  (3): Sequential(
    (0): Linear(in_features=4, out_features=4, bias=True)
    (1): Sigmoid()
  )
  (4): Sequential(
    (0): Linear(in_features=4, out_features=5, bias=True)
    (1): Sigmoid()
  )
  (5): Sequential(
    (0): Linear(in_features=5, out_features=5, bias=True)
    (1): Sigmoid()
  )
  (6): Sequential(
    (0): Linear(in_features=5, out_features=4, bias=True)
    (1): ReLU()
  )
  (7): Sequential(
    (0): Linear(in_features=4, out_features=5, bias=True)
    (1): ReLU()
  )
  (8): Linear(in_features=5, out_features=128, bias=True)
  (9): Linear(in_features=128, out_features=128, bias=True)
  (10): Linear(in_features=128, out_features=2, bias=True)

```

Search space for LSTM autoencoder for anomaly detection:
```yaml
input: [90, 4]
output: [90, 4]
sequence:
  - block: "1"
    op_candidates: ["layer_norm"]
    layer_norm: []
  - block: "2"
    op_candidates: ["lstm"]
    type_repeat:
      type: "repeat_op"
      depth: [ 1]
    lstm:
      hidden_size: [15]
      num_layers: [1, 2]
      bidirectional: [True]
  - block: "3"
    op_candidates: ["gaussian_dropout", "dropout"]
    gaussian_dropout:
      p: [0.1, 0.2]
    dropout:
      p: [ 0.1, 0.2 ]
  - block: "4"
    op_candidates: ["repeat_vector"]
    repeat_vector:
      times: [ 3]
  - block: "5"
    type_repeat:
      type: "repeat_block"
      ref_block: 2
  - block: "6"
    op_candidates: [ "time_distributed_linear" ]
    time_distributed_linear:
      batch_first: [ True ]
      width: [3]
      activation: ["tanh", "sigmoid"]

```

Example Models:
```
Sequential(
  (0): LayerNorm((90, 4), eps=1e-05, elementwise_affine=True)
  (1): SimpleLSTM(
    (lstm): LSTM(4, 15, batch_first=True, bidirectional=True)
  )
  (2): GaussianDropout()
  (3): RepeatVector()
  (4): SimpleLSTM(
    (lstm): LSTM(3, 15, batch_first=True, bidirectional=True)
  )
  (5): TimeDistributed(
    (module): Sequential(
      (0): Linear(in_features=30, out_features=4, bias=True)
      (1): Tanh()
    )
  )
)
Sequential(
  (0): LayerNorm((90, 4), eps=1e-05, elementwise_affine=True)
  (1): SimpleLSTM(
    (lstm): LSTM(4, 15, num_layers=2, batch_first=True, bidirectional=True)
  )
  (2): GaussianDropout()
  (3): RepeatVector()
  (4): SimpleLSTM(
    (lstm): LSTM(3, 15, num_layers=2, batch_first=True, bidirectional=True)
  )
  (5): TimeDistributed(
    (module): Sequential(
      (0): Linear(in_features=30, out_features=4, bias=True)
      (1): Sigmoid()
    )
  )
)
```

# Adding new operations
It is possible to add new operation to the search space with minimal coding effort.
A Layer class must be added in  [layer_builder.py](layer_builder.py).
This class will automatically be added to a layer registry which will be used to look up and translate operations
in the yaml file.
The class must be decorated with ` @register_layer("name_of_op")`  with the string corresponding to the 
operation identifier in the yaml file.
It must implement the following abstract methods:
```python 
class LayerBuilder(ABC):

    @abstractmethod
    def build_layer(self, input_shape, search_parameters: dict):
        pass

    @abstractmethod
    def get_last_layer(self, input_shape, search_parameters: dict, output_shape):
        pass
```
Both must return the instantiated layer and the calculated shape of the output of this layer.
get_last_layer needs to deal with the case that the current layer is the last layer of the network and match its output shape to that.
build_layer deals the the normal case and returns the output shape according to the search parameters.
Both of these methods will be called by the already implemented build method accordingly.

```python 
@register_layer("linear")
class LinearLayer(LayerBuilder):

    def build(self, input_shape, search_parameters: dict, output_shape=None):
        activation = search_parameters.get("activation", None)
        out_shape= output_shape if output_shape is not None else search_parameters["width"]
        linear = nn.Linear(in_features=input_shape, out_features=out_shape)
        next_in_shape= out_shape
        if activation is not None:
            return nn.Sequential(linear, activation_mapping[activation]), next_in_shape
        else:
            return linear, next_in_shape
```

If a new operation type was added, it is possible that the transition to other layers is non-trivial and must be specified
as well.
This can be done in  [layer_adapter.py](layer_adapter.py)
The adapter has to implement forward and infer_output_shape. 
Infer output shape must return the shape output of this adapter.

```python 
class ToLinearAdapter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer= nn.Flatten()

    def forward(self, x):
        return self.layer(x)

    @staticmethod
    def infer_output_shape(input_shape: int | list) -> int | list:
        if isinstance(input_shape, list):
            return math.prod(input_shape)
        else:
            return input_shape
```

After adding and adapter in this way it must be registered in [registry.py](registry.py)
```python 
DEFAULT_ADAPTER = {
    ("conv2d", "lstm"): Conv2dToLSTMAdapter,
    ("linear", "conv2d"): LinearToConv2dAdapter,
    ("linear", "lstm"): LinearToLstmAdapter,
    ("conv2d", "linear"): Conv2dToLinearAdapter,
    ("lstm", "linear"): LSTMNoSequenceAdapter,
    ("lstm", "conv2d"): LSTMToConv2dAdapter,
    ("lstm", None): LSTMNoSequenceAdapter,
    (None, "linear"): ToLinearAdapter,
    ("*", "linear"): ToLinearAdapter,
    ("conv1d", "linear"): ToLinearAdapter,

}
```
The key tuple specifies between which two layer operations this adapter is inserted in the model
and the value specifies the class of the adapter.
If the key contains None this signifies that the adapter should be added at the beginning
or end of the model.
If the adapter should be inserted every time an operation appears, the wildcard * can be used instead of a concrete name.

# Adding new quantization schemes
This works similarly to adding new operations. You implement a QuantizationScheme and corresponding QuantizationBuilder and register it with the @register_quantization_scheme("name_of_scheme") wrapper. For schemes with extra parameters the parse_search_param(...) helper function can be used. 

```python 
class QuantizationScheme(ABC):
    dtype: str

    @staticmethod
    @abstractmethod
    def name() -> str: ...


@dataclass(frozen=True)
class ExampleQuantizationScheme(QuantizationScheme):
    dtype: str = "float16"
    total_bits: int = 2
    @staticmethod
    def name() -> str:
        return "example_quant_scheme"

class QuantizationBuilder(ABC):
    base_type: type[QuantizationScheme]

    def __init__(self, trial, search_params: dict) -> None:
        self.trial = trial
        self.search_params = search_params

    def build(self) -> QuantizationScheme:
        return self.base_type()

@register_quantization_scheme("example_quant_scheme")
class ExampleQuantizationSchemeBuilder(QuantizationBuilder):
    base_type = ExampleQuantizationScheme
    def build(self) -> QuantizationScheme:
        total_bits = parse_search_param(
            self.trial,
            f"total_bits",
            self.search_params,
            "total_bits",
        )
        return self.base_type(total_bits)

```