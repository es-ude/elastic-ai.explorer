# RPi-Generator
The RPi-Generator a implementation of the elastic-ai.Explorer Generator classes. It enables deployment and measurements of hardware metrics on the RPi 4 and 5. The general setup is explained in the top-level [README.md](../../../README.md) under "Set up your Raspberry Pi 4/5 for Deployment".

It recommended to use the explorer.Explorer class to use the Generator. A extensive example of this is found in pi_example.py. 

However, the following is a brief explanation of the usage. 

## Adding new measurement scripts
To adapt the hardware metrics that are measured to your specific task you can add new c++-scripts that can be run on the RPi. Examples for measuring accuracy and latency for models doing MNIST image classification model are found in the [docker folder](../../../docker). 

When registering the source to metric either with the Explorer or with the HWManager directly, don't forget to place your c++-script in the build context given in the CompilerParams. Either then give the absolute path to your script or a relative path to the build context.

Example:
```python
CompilerParams(
    build_context="path/to/buildcontext",
)
metric_to_source = {
    Metric.ACCURACY: Path("relative/path/to/script.cpp")
    }

```
The script needs implement two things to work correctly with the RPi-Generator and communicate the results back to the PC.
First, the tested model is given as a commandline argument by the HW-Manager and can be loaded like this:

```cpp
torch::jit::script::Module module  = torch::jit::load(argv[1]); 
```

After testing the model in someway, the result should be printed to console in a Json format with the following structure {"metric_name": {"value": value, "unit": unit}}

Example: 
```cpp
std::printf(
    "{\"Accuracy\": { \"value\":  %.3f, \"unit\": \"percent\"}}", accuracy_in_percent);
```

## Deploying Test Data
To load data to the RPi specify a deployable dataset path in the DatasetSpecification. It will be copied to the RPi "data"-folder when RPiHWManager.prepare_dataset(...) is called.

```python
DatasetSpecification(
        dataset=YourDataset(root=path_to_test_data, transform=transf),
        deployable_dataset_path=path_to_test_data,
    )
```


## Compiling
The RPICompiler uses docker buildx to cross-compile for the RPi. This is done by layering two docker images. One is the base image which simulates the runtime environment of a RPi and installs all necessary build tool, this is only created one time. The smaller cross-compile image is build on top of the base image and compiles the measurement script on creation. We recommend using Dockerfile.pibase and Dockerfile.picross respectively. These are found in the [docker folder](../../../docker) and can be specified in CompilerParams. The resulting executables are copied to the build context ("build-context/bin").


## Quantization
The RPi Generator currently does not support quantization. 