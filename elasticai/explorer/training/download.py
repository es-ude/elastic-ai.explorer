from abc import ABC, abstractmethod


import os
from pathlib import Path
from typing import Callable, Type, Union

from iesude.data import DataSet
from iesude.data.archives import PlainFile
from iesude.data.extractable import ExtractableFn
import owncloud


class Downloadable(ABC):
    @abstractmethod
    def _download(self, root, file_path_in_sciebo, file_type):
        pass


def get_file_from_sciebo(
    path_to_save: str,
    file_path_in_sciebo: str,
    file_type: ExtractableFn,
):
    if os.path.isfile(path_to_save) or (
        os.path.isdir(path_to_save) and os.listdir(path_to_save)
    ):
        return

    for i in range(10):
        try:
            if file_type is PlainFile:
                dataset = DataSet(file_path=file_path_in_sciebo, file_type=file_type)
                parent = Path(path_to_save).parent
                dataset.download(parent)
                save_path = Path(path_to_save).parent.parent / Path(file_path_in_sciebo)
                os.renames(save_path, path_to_save)

            else:
                dataset = DataSet(file_path=file_path_in_sciebo, file_type=file_type)
                dataset.download(path_to_save)
        except owncloud.HTTPResponseError as err:
            if i < 10:
                continue
            else:
                raise err
        else:
            break


class DownloadableSciebo:
    def __init__(
        self,
        download_path: Union[str, Path],
        file_path_on_sciebo: Union[str, Path],
        file_type: Union[Type, Callable],
        *args,
        **kwargs,
    ) -> None:
        self.download_path = download_path
        self.file_path_on_fileshare = file_path_on_sciebo
        self.file_type = file_type

        self._download()

        super().__init__(*args, **kwargs)

    def _download(self):
        get_file_from_sciebo(
            str(self.download_path), str(self.file_path_on_fileshare), self.file_type
        )
