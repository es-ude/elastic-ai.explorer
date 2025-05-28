# elastic-ai.explorer
HW-NAS-based toolbox for optimizing DNN architectures for different target HW platforms


# install dependencies
Recommended: Use UV as a Package Manager.  https://docs.astral.sh/uv/configuration/installer/

Run following command in project root.
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


# using the docker part
To use first build Dockerfile.cross:
```
 docker buildx build -f Dockerfile.picross --tag cross .
 ```

 then afterwards

 ```
 docker buildx build -f Dockerfile.loader -o type=local,dest=./bin .
  ```

# using crosscompiler for pico

Initialize submodules: 
```
git submodule sync
git submodule update --init --recursive
```
