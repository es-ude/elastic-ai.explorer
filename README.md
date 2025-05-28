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

Build the Explorer-optimized model.cpp for testing:
- Replace the model.cpp in the "pico_crosscompiler/app_\<quantization\>" folder
- From project root do:
```
cd pico_crosscompiler
mkdir build && cd build
cmake .. && make -j4
```
- Flash pico with app_\<quantization\>/app_\<quantization\>.uef2 from build folder

