from elasticai.explorer.hw_nas.search_space.operations import (
    BlockFactory,
    Conv2dBlock,
    LinearBlock,
)


class TestBlockFactory:
    def test_create_convBlock(self):
        block_factory = BlockFactory()
        block = block_factory.createBlock("conv2d", [1, 28, 28], 10)
        assert isinstance(block, Conv2dBlock)
