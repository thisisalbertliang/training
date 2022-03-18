from abc import ABC, abstractmethod
from typing import List

import torch


class DistributedStrategy(ABC):
    """Base class for distributed utilities in PyTorch CUDA and PyTorch/XLA"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_rank(self) -> int:
        """
        Gets distributed rank or returns 0 if distributed is not initialized

        :return: the distributed rank
        :rtype: int
        """
        pass

    @abstractmethod
    def get_world_size(self) -> int:
        """
        Gets distributed world size or returns 1 if distributed is not initialized

        :return: the distributed world size
        :rtype: int
        """
        pass

    @abstractmethod
    def get_device(self, local_rank: int) -> torch.device:
        """
        Sets and gets the backend device associated with the local rank

        :param int local_rank: the local rank for the backend device
        :return: the backend device
        :rtype: torch.device
        """
        pass

    @abstractmethod
    def barrier(self):
        """
        Distributed barrier to synchronize all processes
        """
        pass

    @abstractmethod
    def seed_everything(self, seed: int):
        """
        Seeds random state for backend devices

        :param int seed: the random seed to set
        """
        pass

    @abstractmethod
    def broadcast_seeds(self, seeds: List[int], device: torch.device) -> List[int]:
        """
        Broadcasts the random seeds from the master to all distributed workers

        :param List[int] seeds: list of seeds to broadcast from the master
        :param torch.device device: the backend device
        :return: list of the broadcasted random seeds
        :rtype: List[int]
        """
        pass

    @abstractmethod
    def reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        All-reduces the tensor from all workers using sum operation

        :param torch.Tensor tensor: the tensor to all reduce over
        :return: the all-reduced tensor
        :rtype: torch.Tensor
        """
        pass

    def is_main_process(self) -> bool:
        """
        Returns trues if it is the master process

        :return: true if it is the master process, false otherwise
        :rtype: bool
        """
        return self.get_rank() == 0
