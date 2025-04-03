import unittest
from pathlib import Path
from unittest.mock import Mock

from elasticai.explorer.platforms.deployment.compiler import Compiler
from elasticai.explorer.platforms.deployment.manager import CONTEXT_PATH


class TestCompiler(unittest.TestCase):

    def test_compile_Program(self):

        expected_name_of_executable = "measure_latency"
        path_to_executable = CONTEXT_PATH / "bin" / expected_name_of_executable
        config = Mock(
            compiler_tag="cross",
            path_to_dockerfile=CONTEXT_PATH / "Dockerfile.picross",
            build_context=CONTEXT_PATH,
        )
        compiler = Compiler(config)
        if not compiler.is_setup():
            compiler.setup()
        compiler.compile_code(expected_name_of_executable, "measure_latency.cpp")
        if not Path(path_to_executable).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path_to_executable))


if __name__ == "__main__":
    unittest.main()
