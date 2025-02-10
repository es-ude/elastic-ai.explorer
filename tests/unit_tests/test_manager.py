import unittest
from unittest.mock import Mock

from fabric import Connection

from elasticai.explorer.platforms.deployment.manager import CommandBuilder, PIHWManager


class TestPiHWManager(unittest.TestCase):
    def setUp(self):
        self.hwmanager = PIHWManager()
        self.connection = Mock(Connection)

    def testCommandBuilder(self):
        builder = CommandBuilder("measure_latency")
        builder.add_argument("model_0.pt")
        builder.add_argument("dataset")
        command = builder.build()
        self.assertEqual("./measure_latency model_0.pt dataset", command)

    def test_run_latency_measurements(self):
        self.hwmanager.measure_latency()


if __name__ == '__main__':
    unittest.main()
