import unittest

import torch_xla.core.xla_model as xm
from runtime.distributed.xla_strategy import XLAStrategy


class TestXLAStrategy(unittest.TestCase):
    """Smoke tests for XLAStrategy"""

    def setUp(self):
        """Initializes XLAStrategy object"""
        self.xla_strategy = XLAStrategy()

    def test_get_rank(self):
        """Smoke test for get_rank"""
        self.assertEqual(self.xla_strategy.get_rank(), xm.get_ordinal())

    def test_get_world_size(self):
        """Smoke test for get_world_size"""
        self.assertEqual(self.xla_strategy.get_world_size(), xm.xrt_world_size())

    def test_get_device(self):
        """Smoke test for get_device"""
        self.assertEqual(self.xla_strategy.get_device(0), xm.xla_device())

    def test_seed_everything(self):
        """Smoke test for seed_everything"""
        seed = 1
        self.xla_strategy.seed_everything(seed)
        self.assertEqual(xm.get_rng_state(), seed)

    def tearDown(self):
        del self.xla_strategy


if __name__ == "__main__":
    unittest.main()
