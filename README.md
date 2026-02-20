# Elastic-Ai.Explorer
HW-NAS-based toolbox for optimizing DNN architectures for different target HW platforms, automated deployment and testing.
Currently supported are the **Raspberry Pi 4/5** and the **Raspberry Pi Pico**. 

This project is still in active development and has no official release yet.

# Install Dependencies
Recommended:
Use **UV** as a Package Manager (https://docs.astral.sh/uv/configuration/installer/)

Then Run following command in project root. 

For Linux Users run:

```
 uv sync --all-groups
 ```
For Mac:

```
 uv sync --no-group nomac
 ```

If you don't need dev dependencies add:

```
 --no-dev
 ```

# Setup for Deployment and System Tests:
 To compile for deployment on hardware you need to install:

### Either:
- Docker-Desktop (https://docs.docker.com/desktop/)

### Or:
- The Docker Engine (https://docs.docker.com/engine/install/)
- And the Docker-Buildx-Plugin (https://github.com/docker/buildx)

First Deployment with Docker for each platform is slow because the necessary Docker-Images have to be created, after that the Docker-Images can be reused and deployment is much faster. 

## Setup your Raspberry Pi 4/5 for Deployment
To use the Explorer to deploy models on your Raspberry Pi, we recommend using Bookworm as an OS. You also need to enable ssh connections on your RPi and make one initial connection between your host PC and the RPi.  
Then install libtorch on your Pi under "/code/libtorch" directly at the root of your system, add this libtorch version also under "docker/code/libtorch" in the elastic-ai.Explorer. You can find precompiled versions of libtorch for Bookworm on RPi4 and RPi5 here (https://uni-duisburg-essen.sciebo.de/s/9aiYf5Y2NABtdQb).

After this you can use the System Tests by creating your own system_test_settings.toml as shown in "tests/system_tests/example_system_test_settings.toml". Similarly, you can use the example ("examples/pi_example.py") by adding your RPi's credentials to the SSHParams. 

## Setup your Raspberry Pi Pico for Deployment
There should be no setup on device necessary, just connect the Pico with your host PC and find the correct device path (on Linux probably "media/RPI-RP2"). Additionally it can be necessary to add the user to dialout and tty group at the serial port (default is "/dev/ttyACM0") in order to communicate over the serial connection.

After this you can use the System Tests for Pico by creating your own system_test_settings.toml as shown in "tests/system_tests/example_system_test_settings.toml". Similarly, you can use the example ("examples/pico_example.py") by adding your device path and serial port to the SerialParams. 

## Examples 
For the full workflow from HW-NAS to on-device measurements, see the examples folder.

To configure the Explorer for your specific setup, create your own OptimizationCriteriaRegistry and add your objectives, soft constraints, and hard constraints linked to the estimates provided by the Estimators. Additionally, you can set search strategies and search parameters to further configure your search.

For test deployment and hardware-specific search, create your own HWPlatform with a Generator, Compiler, Host, and HwManager. You can also use the out-of-the-box solutions shown in the examples, or write your own classes using the provided interfaces.