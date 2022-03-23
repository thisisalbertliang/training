from typing import List

import torch
import torch_xla.core.xla_model as xm
from runtime.distributed.distributed_strategy import DistributedStrategy


class XLAStrategy(DistributedStrategy):
    """Distributed utilities for PyTorch/XLA"""

    def __init__(self) -> None:
        super().__init__()
        world_size = self.get_world_size()
        if xm.is_master_ordinal():
            print(f"torch-xla distributed initialized. World size: {world_size}")

    def get_rank(self) -> int:
        """Overrides DistributedStrategy.get_rank"""
        return xm.get_ordinal()

    def get_world_size(self) -> int:
        """Overrides DistributedStrategy.get_world_size"""
        return xm.xrt_world_size()

    def get_device(self, local_rank: int) -> torch.device:
        """Overrides DistributedStrategy.get_device"""
        return xm.xla_device()

    def barrier(self):
        """Overrides DistributedStrategy.barrier"""
        xm.rendezvous("barrier")

    def seed_everything(self, seed: int):
        """Overrides DistributedStrategy.seed_everything"""
        xm.set_rng_state(seed)

    def broadcast_seeds(self, seeds: List[int], device: torch.device) -> List[int]:
        """Overrides DistributedStrategy.broadcast_seeds"""
        if self.get_world_size() > 1:
            seeds_tensor = torch.LongTensor(seeds).to(device)
            self._broadcast(seeds_tensor, 0)
            seeds = seeds_tensor.tolist()
        return seeds

    def reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Overrides DistributedStrategy.reduce_tensor"""
        world_size = self.get_world_size()
        if world_size > 1:
            rt = tensor.clone()
            rt = xm.all_reduce(
                xm.REDUCE_SUM,
                rt,
            )
            if rt.is_floating_point():
                rt = rt / world_size
            else:
                rt = rt // world_size
            return rt
        return tensor

    def _broadcast(self, tensor: torch.Tensor, source_rank: int):
        # only broadcast when there are more than one process
        if self.get_world_size() > 1:
            # fill the worker tensors with zeros
            if self.get_rank() != source_rank:
                tensor.fill_(0)
            # since only the master tensor is non-zero,
            # all-reduce is equivalent to broadcast
            xm.all_reduce(
                reduce_type=xm.REDUCE_SUM,
                inputs=tensor,
            )
