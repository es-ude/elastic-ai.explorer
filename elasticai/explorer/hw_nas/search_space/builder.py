class ComponentBuilder:
    base_type: type | None = None

    def __init__(self, trial, block: dict, search_params: dict, block_id):
        if self.base_type is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define base_type"
            )
        self.trial = trial
        self.block = block
        self.search_params = search_params
        self.block_id = block_id
