import os
from abc import ABC, abstractmethod

from fabric.testing.fixtures import connection
from python_on_whales import docker
from fabric import Connection

CONTEXT_PATH="../../../../docker"

class HWManager(ABC):
    @abstractmethod
    def deploy(self, model):
        pass

class PIHWManager(HWManager):

    def __init__(self):

       if not docker.images("cross"):
           self.build_compiler()

    def deploy(self, model):
        pass

    def build_compiler(self):
        docker.build(CONTEXT_PATH, file=CONTEXT_PATH+"/Dockerfile.picross", tags="cross")



    def compile_code(self):
        docker.build(CONTEXT_PATH, file=CONTEXT_PATH+"/Dockerfile.loader", output={"type": "local", "dest": CONTEXT_PATH+"/bin"})



    def deploy_model_on_pi(self):
        ...
def connect_to_pi():
    connection= Connection(host="transpi5.local", user="ies")
    print(connection.put("../../../../docker/bin/measure_latency"))
    print(connection.put("../../../../models/ts_models/model_0.pt"))
    result = Connection(host="transpi5.local", user="ies").run('./measure_latency model_0.pt', hide=False)

    msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"

    print(msg.format(result))
if __name__ == '__main__':
    manager = PIHWManager()
    manager.compile_code()
    connect_to_pi()
   # manager.deploy_on_pi()