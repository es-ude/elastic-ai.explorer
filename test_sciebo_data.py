from iesude.data import DataSet
from iesude.data.archives import Zip

MyNewDataSet = DataSet(file_path="Transfair/mnist.zip", file_type=Zip)

MyNewDataSet.download("data")
