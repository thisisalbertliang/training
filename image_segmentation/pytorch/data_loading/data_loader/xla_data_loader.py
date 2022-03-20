from argparse import Namespace
from typing import Tuple

import torch
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def get_data_loaders(
    flags: Namespace,
    num_shards: int,
    global_rank: int,
    device: torch.device,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> Tuple[pl.MpDeviceLoader, pl.MpDeviceLoader]:
    """
    Initializes and returns (train_data_loader, val_data_loader) as MpDeviceLoader objects

    :param Namespace flags: the runtime arguments
    :param int num_shards: number of shards for the train dataset
    :param int global_rank: global rank associated with the device
    :param torch.device device: the device to use for MpDeviceLoader
    :param Dataset train_dataset: the train dataset
    :param Dataset val_dataset: the validation dataset
    :return: the tuple (train_loader, val_loader)
    :rtype: Tuple[pl.MpDeviceLoader, pl.MpDeviceLoader]
    """
    if num_shards > 1:
        # train sampler for data parallelism
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_shards,
            rank=global_rank,
            seed=flags.seed,
            drop_last=True,
        )
    else:
        train_sampler = None

    val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=flags.batch_size,
        shuffle=not flags.benchmark and train_sampler is None,
        sampler=train_sampler,
        num_workers=flags.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=not flags.benchmark and val_sampler is None,
        sampler=val_sampler,
        num_workers=flags.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    train_loader = pl.MpDeviceLoader(train_loader, device)
    val_loader = pl.MpDeviceLoader(val_loader, device)

    return train_loader, val_loader
