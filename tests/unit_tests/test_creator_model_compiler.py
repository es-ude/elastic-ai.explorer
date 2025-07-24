from pathlib import Path

from elasticai_explorer.generator.model_compiler.model_compiler import CreatorModelCompiler
from tests.unit_tests.samples.sample_MLP import SimpleSampleMLP


class TestCreatorModelCompiler:
    def setup_method(self) -> None:
        self.mlp = SimpleSampleMLP(4)
    def test_compile_module_to_ir(self):
        compiler = CreatorModelCompiler()
        compiler.generate(model=self.mlp, path=Path(""))
