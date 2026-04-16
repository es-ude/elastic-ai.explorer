# RPI-Generator
The RPI-Generator is an implementation of the elastic-ai.Explorer Generator classes. It enables deployment and measurements of hardware metrics on the RPi 4 and 5. The general setup is explained in the top-level [README.md](../../../README.md) under "Set up your Raspberry Pi 4/5 for Deployment".

A extensive example of how to use the Generator after the setup is found in pi_example.py. 

However, the following is a brief explanation of the usage. 

## Adding new measuring scripts
To adapt the hardware metrics that are measured to your specific task you can add new c++-scripts. Examples for measuring accuracy and latency for models doing MNIST image classification model are found in the [docker folder](../../../docker). 

When registering the source to metric either with the Explorer or with the HWManager directly, don't forget to place your c++-script in the build context given in the CompilerParams. Either then give the absolute path to your script or relative to the build context.

Example:
```python
CompilerParams(
    build_context="path/to/buildcontext",
)
metric_to_source = {
    Metric.ACCURACY: Path("relative/path/to/script.cpp")
    }

```
The script needs to a specific behavior to communicate the results back to the PC. 
The result should be printed to console in a Json format with the following structure {"metric_name": {"value": value, "unit": unit}}

Example: 
```cpp
std::printf(
    "{\"Accuracy\": { \"value\":  %.3f, \"unit\": \"percent\"}}", accuracy_in_percent);
```

## Compiling
The RPICompiler uses docker buildx to cross-compile for the RPi. This is done by layering two docker images. One is the base image which simulates the runtime environment of a RPi and installs all necessary build tool, this is only created one time. The smaller cross-compile image is build on top of the base image and compiles the measurement script on creation. We recommend using Dockerfile.pibase and Dockerfile.picross respectively. These are found in the [docker folder](../../../docker) and can be specified in CompilerParams. The resulting executables are copied to the build context ("build-context/bin").
