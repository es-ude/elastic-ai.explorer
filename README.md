# Elastic-Ai.Explorer
HW-NAS-based toolbox for optimizing DNN architectures for different target HW platforms, automated deployment and testing.
Currently supported are the **Raspberry Pi 4/5** and the **Raspberry Pi Pico**. Additionally the **Elastic Node v5** is supported in simulation and deployment (experimental). 

This project is still in active development and has no official release yet. As stated above three platforms (RPi4/5, RPi Pico) are already supported. The implementations of the corresponding generator classes are located in elasticai.explorer-impl.

# Install Dependencies
Recommended:
Use **UV** as a Package Manager (https://docs.astral.sh/uv/configuration/installer/)

Then Run following command in project root. 

## Linux and Mac
Run:

```
uv sync
 ```

If you don't need dev dependencies:

```
uv sync --no-dev
 ```

## Windows
#### TODO check compatibility without Pico-Generator
For easy setup on Windows use the Dockerfile.explorer to create an image:
```
docker build -t explorer -f Dockerfile.explorer . 
 ```
If used for cross-compilation with docker, run the Image with "-v /var/run/docker.sock:/var/run/docker.sock" argument to allow for creating sibling docker images.

```
docker run -v /var/run/docker.sock:/var/run/docker.sock
 ```

## Pico-Generator Dependencies
If you want to use the pico-generator, install the additional dependencies with:

```
uv sync --extra pico-generator
 ```

When using the Pico-Generator in an other Project and you do not want to allow pre-releases, add:
```
[tool.uv]
exclude-dependencies = ["tf-nightly"]
...
```

## Creator Generator 
If you want to use the creator-generator, install the additional dependencies with:
```
uv sync --extra creator-generator
 ```

To simulate the deployment of neural networks on the ENv5 and the builtin Spartan 7 FPGA, we use cocotb and GHDL. GHDL>=5.1.1 has to be installed manually, since it isn't a python package (https://github.com/ghdl/ghdl?tab=readme-ov-file#getting-ghdl).

# Setup for Deployment and System Tests:
 To compile for deployment on hardware you need to install:

### Either:
- Docker-Desktop (https://docs.docker.com/desktop/)

### Or:
- The Docker Engine (https://docs.docker.com/engine/install/)
- And the Docker-Buildx-Plugin (https://github.com/docker/buildx)

>First Deployment with Docker for each platform is slow because the necessary Docker-Images have to be created, after that the Docker-Images can be reused and deployment is much faster. To speed up the setup significantly you can also download the base images [here](https://uni-duisburg-essen.sciebo.de/s/9aiYf5Y2NABtdQb).

### Troubleshooting

To resolve the following error:
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```
Follow the steps in the Docker documentation: [How do I use Docker SDKs with Docker Desktop for Linux?](https://docs.docker.com/desktop/troubleshoot-and-support/faqs/linuxfaqs/#how-do-i-use-docker-sdks-with-docker-desktop-for-linux)

## Set up your Raspberry Pi 4/5 for Deployment
To use the Explorer to deploy models on your Raspberry Pi, we recommend using Bookworm 64-Bit as an OS. You also need to enable ssh connections on your RPi and make one initial connection between your host PC and the RPi. Make sure to add you public SSH key to the Raspberry Pi under `~/.ssh/authorized_keys`.

Then install libtorch on your Pi under `/code/libtorch` directly at the root of your system, add this libtorch version also under the same path relative to the docker build context (this should be `docker/code/libtorch`). You can find precompiled versions of libtorch for Bookworm on RPi4 and RPi5 [here](https://uni-duisburg-essen.sciebo.de/s/9aiYf5Y2NABtdQb).

Ensure a `data` directory exists on the Raspberry Pi user's home directory.
More information on how to use the Pico-Generator implementation is found [here](elasticai/explorer_impl/rpi_generator/README.md).

After this you can use the System Tests by creating your own system_test_settings.toml as shown in example_system_test_settings.toml in the system test folder. Similarly, you can use the example (pi_example.py) by adding your RPi's credentials to the SSHParams. 

## Set up your Raspberry Pi Pico / Pico2 for Deployment
There should be no setup on device necessary, just connect the Pico with your host PC and find the correct device path (on Linux probably `/media/RPI-RP2` for Pico or `/media/RP2350` for Pico2). Additionally, it can be necessary to add the user to dialout and tty group at the serial port (default is `/dev/ttyACM0`) in order to communicate over the serial connection.
Importantly, do not forget to install the additional dependencies for the pico-generator as explained in [Pico-Generator Dependencies](#Pico-Generator-Dependencies).

After this you can use the System Tests for Pico by creating your own system_test_settings.toml as shown in example_system_test_settings.toml in the system test folder. Similarly, you can use the example (pico_example.py) by adding your device path and serial port to the SerialParams. 

To change the Compiler from the Pico to the Pico2 simply add `additional_params= {"platform_type": "rp2350"}` to your CompilerParams and create a new base image (give it a new name like "pico2base" in CompilerParams). 

To use the deployment pipeline for the Pico, it relies on a docker based cross-compiler. As an example, you can use the pico_crosscompiler from `docker/code/pico_crosscompiler` and the dockerfiles picobase and picocross under `docker`. Adapt this to your use case and set it in CompilerParams. More on this [here](elasticai/explorer_impl/pico_generator/README.md).

## Set up your ENv5 for Deployment
In the current version the Deployment Pipeline on the ENv5 is not completely stable. If you still want to try the deployment, you have to flash your ENv5 with the HardwareTestUsbProtocol.uf2 from the [elastic-ai.runtime.enV5](https://github.com/es-ude/elastic-ai.runtime.enV5). Build the HardwareTestUsbProtocol with cmake manually or use devenv with preconfigured build tasks, for more on this see the runtime [Readme](https://github.com/es-ude/elastic-ai.runtime.enV5).

The ENv5 uses a RP2040 MCU (same as RPi Pico) for loading models and data on the FPGA, therefore the general setup of SerialParams is the same. An exception is the baud rate (baud_rate=9600), for more on this see the `env5_example.py`. 


## Examples 
For the full workflow from HW-NAS to on-device measurements, see the examples folder.

To configure the Explorer for your specific setup, create your own OptimizationCriteriaRegistry and add your objectives, soft constraints, and hard constraints linked to the estimates provided by the Estimators. Additionally, you can set search strategies and search parameters to further configure your search.

For test deployment and hardware-specific search, create your own Generator with a ModelBuilder, ModelTranslator, Host, and HwManager. You can also use the out-of-the-box solutions shown in the examples, or write your own classes using the provided interfaces.

# Search Space Specification
To learn how to specify your own search space for a HW-NAS in YAML format or to learn how to extend the supported operations in code
see the [search space specification](elasticai/explorer/hw_nas/search_space/README.md).

# Generator Implementations
More information on how to use the generators and build your own Generator can be found here in [Generator](elasticai/explorer/generator/README.md).