import random
from functools import wraps
from typing import List, Optional, Tuple

import numpy as np
import torch
from runtime.distributed.cuda_strategy import CUDAStrategy
from runtime.distributed.distributed_strategy import DistributedStrategy
from runtime.distributed.xla_strategy import XLAStrategy

_STRATEGY: Optional[DistributedStrategy] = None


def init_distributed(flags) -> bool:
    """Initializes the distributed backends (either PyTorch CUDA or PyTorch/XLA)

    :param flags: the runtime arguments
    :return: false if using single core training, true otherwise
    :rtype: bool
    """
    global _STRATEGY
    if flags.device == "xla":
        _STRATEGY = XLAStrategy()
    elif flags.device == "cuda":
        _STRATEGY = CUDAStrategy()
    else:
        raise ValueError(f"Device {flags.device} unknown. Valid devices are: cuda, xla")
    return get_world_size() > 1


def _assert_initialized(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        assert _STRATEGY is not None, (
            "Please initialize the distributed_utils module first "
            "by running `init_distributed(flags)`"
        )
        return func(*args, **kwargs)

    return wrapper


@_assert_initialized
def get_device(local_rank: int) -> torch.device:
    """Sets and gets the backend device associated with the local rank

    :param int local_rank: the local rank for the backend device
    :return: the backend device
    :rtype: torch.device
    """
    return _STRATEGY.get_device(local_rank)


@_assert_initialized
def seed_everything(seed: int):
    """Seeds random state for torch, numpy, and backend devices

    :param int seed: the random seed to set
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return _STRATEGY.seed_everything(seed)


def generate_seeds(rng: random.Random, size: int) -> List[int]:
    """Generates list of random seeds

    :param random.Random rng: random number generator
    :param int size: length of the returned list
    :return: list of random seeds
    :rtype: List[int]
    """
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds


@_assert_initialized
def broadcast_seeds(seeds: List[int], device: torch.device) -> List[int]:
    """Broadcasts the random seeds from the master to all distributed workers

    :param List[int] seeds: list of seeds to broadcast from the master
    :param torch.device device: the backend device
    :return: list of the broadcasted random seeds
    :rtype: List[int]
    """
    return _STRATEGY.broadcast_seeds(seeds, device)


def setup_seeds(
    epochs: int, device: torch.device, master_seed: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """Generates seeds from one master_seed.

    Function returns (worker_seeds, shuffling_seeds), worker_seeds are later
    used to initialize per-worker random number generators (mostly for
    dropouts), shuffling_seeds are for RNGs resposible for reshuffling the
    dataset before each epoch.
    Seeds are generated on worker with rank 0 and broadcasted to all other
    workers.

    :param Optional[int] master_seed: master RNG seed used to initialize other generators
        if None, a random master_seed is generated
    :param int epochs: number of epochs
    :param torch.device device: backend device used for distributed broadcast
    :return: (worker_seeds, shuffling_seeds)
    :rtype: Tuple[List[int], List[int]]
    """
    if master_seed is None:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2**32 - 1)
        if get_rank() == 0:
            # master seed is reported only from rank=0 worker, it's to avoid
            # confusion, seeds from rank=0 are later broadcasted to other
            # workers
            print(f"Using random master seed: {master_seed}")
    else:
        # master seed was specified from command line
        print(f"Using master seed from command line: {master_seed}")

    # initialize seeding RNG
    seeding_rng = random.Random(master_seed)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(seeding_rng, get_world_size())

    # generate seeds for data shuffling, one seed for every epoch
    shuffling_seeds = generate_seeds(seeding_rng, epochs)

    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device)
    shuffling_seeds = broadcast_seeds(shuffling_seeds, device)
    return worker_seeds, shuffling_seeds


@_assert_initialized
def get_world_size() -> int:
    """Gets distributed world size or returns 1 if distributed is not initialized

    :return: the distributed world size
    :rtype: int
    """
    return _STRATEGY.get_world_size()


@_assert_initialized
def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduces the tensor from all workers using sum operation

    :param torch.Tensor tensor: the tensor to all reduce over
    :return: the all-reduced tensor
    :rtype: torch.Tensor
    """
    return _STRATEGY.reduce_tensor(tensor)


@_assert_initialized
def get_rank() -> int:
    """Gets distributed rank or returns 0 if distributed is not initialized

    :return: the distributed rank
    :rtype: int
    """
    return _STRATEGY.get_rank()


@_assert_initialized
def is_main_process() -> bool:
    """Returns trues if it is the master process

    :return: true if it is the master process, false otherwise
    :rtype: bool
    """
    return _STRATEGY.is_main_process()


@_assert_initialized
def barrier():
    """Distributed barrier to synchronize all processes"""
    return _STRATEGY.barrier()
