from argparse import Namespace
from typing import Tuple

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def get_data_loaders(
    flags: Namespace,
    num_shards: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> Tuple[DataLoader, DataLoader]:
    """Initializes and returns (train_data_loader, val_data_loader) as Dataloader objects

    :param Namespace flags: the runtime arguments
    :param int num_shards: number of shards for the train dataset
    :param Dataset train_dataset: the train dataset
    :param Dataset val_dataset: the validation dataset
    :return: the tuple (train_loader, val_loader)
    :rtype: Tuple[DataLoader, DataLoader]
    """
    if num_shards > 1:
        # train sampler for data parallelism
        train_sampler = DistributedSampler(train_dataset, seed=flags.seed, drop_last=True)
    else:
        train_sampler = None

    val_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=flags.batch_size,
        shuffle=not flags.benchmark and train_sampler is None,
        sampler=train_sampler,
        num_workers=flags.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=not flags.benchmark and val_sampler is None,
        sampler=val_sampler,
        num_workers=flags.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, val_dataloader
