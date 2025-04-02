from iesude.data import DataSet
from iesude.data.archives import Zip, PlainFile

mnist_dataset = DataSet(file_path="mnist.zip", file_type=PlainFile)

mnist_dataset.download("data/test")



# from iesude.data import MitBihAtrialFibrillationDataSet as AFDataSet

# AFDataSet.download("data")
#  DataSet(
#     file_path="mit-bih-atrial-fibrillation.tar", file_type=TarArchive
# )