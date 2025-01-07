# elastic-ai.explorer
HW-NAS-based toolbox for optimizing DNN architectures for different target HW platforms

# using the docker part
To use first build Dockerfile.cross:
```
 docker buildx build -f Dockerfile.picross --tag cross .
 ```

 then afterwards

 ```
 docker buildx build -f Dockerfile.loader -o type=local,dest=./bin .
  ```