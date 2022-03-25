import unittest
from argparse import Namespace

import data_loading.data_loader.xla_data_loader as xl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from data_loading.data_loader.unet3d_data_loader import SyntheticDataset


class TestXLADataLoader(unittest.TestCase):
    """Smoke test for data_loading.data_loader.xla_data_loader"""

    def setUp(self):
        """Initializes arguments to get_data_loaders"""
        self.flags = Namespace(
            seed=0,
            batch_size=1,
            benchmark=False,
            num_workers=4,
            input_shape=(128, 128, 128),
            layout="NCDHW",
        )
        self.num_shards = 1
        self.global_rank = 0
        self.device = xm.xla_device()
        self.train_dataset = SyntheticDataset(
            scalar=True,
            shape=self.flags.input_shape,
            layout=self.flags.layout,
        )
        self.val_dataset = SyntheticDataset(
            scalar=True,
            shape=self.flags.input_shape,
            layout=self.flags.layout,
        )

    def test_get_data_loaders(self):
        """Smoke test for get_data_loaders"""
        train_loader, val_loader = xl.get_data_loaders(
            flags=self.flags,
            num_shards=self.num_shards,
            global_rank=self.global_rank,
            device=self.device,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
        )

        self.assertTrue(isinstance(train_loader, pl.MpDeviceLoader))
        self.assertTrue(isinstance(val_loader, pl.MpDeviceLoader))

        self.assertEqual(train_loader._device, self.device)
        self.assertEqual(val_loader._device, self.device)

        self.assertEqual(len(train_loader), 64)
        self.assertEqual(len(val_loader), 64)
