from abc import ABC, abstractmethod
from python_on_whales import docker

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



    def deploy_on_pi(self):
       ...

if __name__ == '__main__':
    manager=PIHWManager()
    manager.compile_code()
   # manager.deploy_on_pi()