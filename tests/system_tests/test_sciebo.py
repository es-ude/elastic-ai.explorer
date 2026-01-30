from elasticai.explorer.training.download import get_file_from_sciebo

from iesude.data.archives import Zip, PlainFile
import shutil
import os

from settings import ROOT_DIR


class TestSciebo:
    def setup_class(self):
        self.save_dir = str(ROOT_DIR / "tests/system_tests/samples/data")
        
    def test_sciebo_download(self):

        get_file_from_sciebo(
            path_to_save= self.save_dir + "/sciebo",
            file_path_in_sciebo="mnist.zip",
            file_type=Zip,
        )
        assert os.path.isdir(self.save_dir + "/sciebo")

    def test_sciebo_Plainfile(self):
        get_file_from_sciebo(
            path_to_save=self.save_dir + "/test_data.csv",
            file_path_in_sciebo="test_dataset.csv",
            file_type=PlainFile,  # type: ignore
        )
        assert os.path.isfile(self.save_dir + "/test_data.csv")

    def teardown_class(self):
        shutil.rmtree(self.save_dir, ignore_errors=True)