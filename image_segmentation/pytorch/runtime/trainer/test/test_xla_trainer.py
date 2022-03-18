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
        flags = Namespace(
            amp=False,
            ga_steps=1,
            layout="NCDHW",
            include_background=False,
            batch_size=self.batch_size,
            input_shape=self.image_size,
            val_input_shape=self.image_size,
            benchmark=False,
            num_workers=8,
            optimizer="sgd",
            learning_rate=0.8,
            momentum=0.9,
            weight_decay=0.0,
            lr_decay_epochs=[],
            lr_decay_factor=1.0,
            loader="synthetic",
            torch_xla=True,
        )

        init_distributed(flags)

        model = Unet3D(
            1,
            3,
            normalization="instancenorm",
            activation="relu",
        )

        self.device = xm.xla_device()
        train_loader, val_loader = get_data_loaders(
            flags=flags,
            num_shards=1,
            global_rank=0,
            device=self.device,
        )

        loss_fn = DiceCELoss(
            to_onehot_y=True,
            use_softmax=True,
            layout=flags.layout,
            include_background=flags.include_background,
        )
        score_fn = DiceScore(
            to_onehot_y=True,
            use_argmax=True,
            layout=flags.layout,
            include_background=flags.include_background,
        )

        self.xla_trainer = XLATrainer(
            flags=flags,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            score_fn=score_fn,
            device=self.device,
            callbacks=[],
        )

        input_size = (self.batch_size, 1) + self.image_size

        self.image = torch.zeros(size=input_size, device=self.device)
        self.label = torch.zeros_like(self.image, device=self.device)

    def test_forward_pass(self):
        """Smoke test for forward pass"""
        loss_value = self.xla_trainer.forward_pass(self.image, self.label)

        self.assertEqual(loss_value.size(), torch.Size([]))

    def test_backward_pass(self):
        """Smoke test for backward pass"""
        loss_value = self.xla_trainer.forward_pass(self.image, self.label)
        self.xla_trainer.backward_pass(iteration=0, loss_value=loss_value)
