import unittest
from argparse import Namespace

import torch
import torch_xla.core.xla_model as xm
from data_loading.data_loader.unet3d_data_loader import get_data_loaders
from model.losses import DiceCELoss, DiceScore
from model.unet3d import Unet3D
from runtime.distributed.distributed_utils import init_distributed
from runtime.trainer.xla_trainer import XLATrainer


class TestXLATrainer(unittest.TestCase):
    """Smoke tests for XLATrainer"""

    def setUp(self):
        """Initializes XLATrainer object"""
        self.batch_size = 1
        self.image_size = (128, 128, 128)
        self.flags = Namespace(
            amp=False,
            ga_steps=1,
            layout="NCDHW",
            include_background=False,
            batch_size=self.batch_size,
            input_shape=self.image_size,
            val_input_shape=self.image_size,
            benchmark=False,
            optimizer="sgd",
            learning_rate=0.8,
            momentum=0.9,
            weight_decay=0.0,
            lr_decay_epochs=[],
            lr_decay_factor=1.0,
            loader="synthetic",
            device="xla",
        )

        init_distributed(self.flags)

        self.device = xm.xla_device()
        self.model = Unet3D(
            1,
            3,
            normalization="instancenorm",
            activation="relu",
        )
        self.loss_fn = DiceCELoss(
            to_onehot_y=True,
            use_softmax=True,
            layout=self.flags.layout,
            include_background=self.flags.include_background,
        )
        self.score_fn = DiceScore(
            to_onehot_y=True,
            use_argmax=True,
            layout=self.flags.layout,
            include_background=self.flags.include_background,
        )
        self.input_size = (self.batch_size, 1) + self.image_size

    def test_train_step(self):
        """Smoke test for the forward pass, backward pass, and model weights update"""
        for num_workers in (1, 8):
            self.flags.num_workers = num_workers

            train_loader, val_loader = get_data_loaders(
                flags=self.flags,
                num_shards=1,
                global_rank=0,
                device=self.device,
            )

            xla_trainer = XLATrainer(
                flags=self.flags,
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=self.loss_fn,
                score_fn=self.score_fn,
                device=self.device,
                callbacks=[],
            )

            images = torch.zeros(size=self.input_size, device=self.device)
            labels = torch.zeros_like(images, device=self.device)

            loss_value = xla_trainer.train_step(iteration=0, images=images, labels=labels)

            self.assertEqual(loss_value.size(), torch.Size([]))
