import os
from typing import List

import torch
import torch.distributed
from runtime.distributed.distributed_strategy import DistributedStrategy


class CUDAStrategy(DistributedStrategy):
    """Distributed utilities for PyTorch CUDA"""

    def __init__(self) -> None:
        super().__init__()
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        distributed = world_size > 1
        if distributed:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend, init_method="env://")
            assert torch.distributed.is_initialized()

        if self.get_rank() == 0:
            print("Distributed initialized. World size:", world_size)

    def get_rank(self) -> int:
        """Overrides DistributedStrategy.get_rank"""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        return rank

    def get_world_size(self) -> int:
        """Overrides DistributedStrategy.get_world_size"""
        return int(os.environ.get("WORLD_SIZE", 1))

    def get_device(self, local_rank: int) -> torch.device:
        """Overrides DistributedStrategy.get_device"""
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    def barrier(self):
        """
        Overrides DistributedStrategy.barrier

        Currently, pytorch doesn't implement barrier for NCCL backend.
        Thus, we call all_reduce on dummy tensor and synchronizes with GPU.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
            torch.cuda.synchronize()

    def seed_everything(self, seed: int):
        """Overrides DistributedStrategy.seed_everything"""
        torch.cuda.manual_seed_all(seed)

    def broadcast_seeds(self, seeds: List[int], device: torch.device) -> List[int]:
        """Overrides DistributedStrategy.broadcast_seeds"""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            seeds_tensor = torch.LongTensor(seeds).to(device)
            torch.distributed.broadcast(seeds_tensor, 0)
            seeds = seeds_tensor.tolist()
        return seeds

    def reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Overrides DistributedStrategy.reduce_tensor"""
        world_size = self.get_world_size()
        if world_size > 1:
            rt = tensor.clone()
            torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
            if rt.is_floating_point():
                rt = rt / world_size
            else:
                rt = rt // world_size
            return rt
        return tensor
