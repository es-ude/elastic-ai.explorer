from abc import ABC, abstractmethod
from elasticai.explorer.training.data import get_file_from_sciebo


import os
from pathlib import Path
from typing import Callable, Type, Union


class Downloadable(ABC):
    @abstractmethod
    def _download(self, root, file_path_in_sciebo, file_type):
        pass


class DownloadableSciebo:
    def __init__(
        self,
        download: bool,
        download_path: Union[str, Path],
        file_path_on_sciebo: Union[str, Path],
        file_type: Union[Type, Callable],
        *args,
        **kwargs,
    ) -> None:
        self.download_path = download_path
        self.file_path_on_fileshare = file_path_on_sciebo
        self.file_type = file_type

        if download:
            self._download()

        super().__init__(*args, **kwargs)

    def _download(self):
        get_file_from_sciebo(
            str(self.download_path), str(self.file_path_on_fileshare), self.file_type
        )
