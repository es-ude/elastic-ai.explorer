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

## Additionally:
 To compile for deployment on hardware you need to install:

### Either:
- Docker-Desktop (https://docs.docker.com/desktop/)

### Or:
- The Docker Engine (https://docs.docker.com/engine/install/)
- And the Docker-Buildx-Plugin (https://github.com/docker/buildx)

## Examples 
For the full workflow from HW-NAS to on-device tests, see the examples folder.

To configure the Explorer for your specific setup, create your own OptimizationCriteriaRegistry and add your objectives, soft constraints, and hard constraints linked to the estimates provided by the Estimators. Additionally, you can set search strategies and search parameters to further configure your search.

For test deployment and hardware-specific search, create your own HWPlatform with a Generator, Compiler, Host, and HwManager. You can also use the out-of-the-box solutions shown in the examples, or write your own classes using the provided interfaces.