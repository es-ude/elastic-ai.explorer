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

# Examples 
For the full workflow from HW-NAS to Tests on Device, see the "examples" Folder.
To configure the example for your specific setup, create your own "deployment_config.yaml" with the help of the templates in "config_files/config_templates".