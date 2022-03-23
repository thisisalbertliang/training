import torch
import torch.backends.cudnn
import torch.cuda.amp
import torch.distributed
import torch.utils.data
import torch.utils.data.distributed
from runtime.distributed.distributed_utils import get_world_size
from runtime.trainer.unet3d_trainer import UNet3DTrainer


class CUDATrainer(UNet3DTrainer):
    """Trains UNet3D in PyTorch CUDA"""

    def __init__(
        self,
        flags,
        model,
        train_loader,
        val_loader,
        loss_fn,
        score_fn,
        device,
        callbacks,
    ) -> None:
        super().__init__(
            flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks
        )

        torch.backends.cudnn.benchmark = flags.cudnn_benchmark
        torch.backends.cudnn.deterministic = flags.cudnn_deterministic

        # Setup model for distributed data parallel
        if get_world_size() > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[flags.local_rank],
                output_device=flags.local_rank,
            )

        # Setup grad scaler and autocast
        self.scaler = torch.cuda.amp.GradScaler()

        # Setup train sampler
        self.train_sampler = self.train_loader.sampler

    def forward_pass(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Overrides UNet3DTrainer.forward_pass"""
        with torch.cuda.amp.autocast(enabled=self.flags.amp):
            output = self.model(images)
            loss_value = self.loss_fn(output, labels)
            loss_value /= self.flags.ga_steps
        return loss_value

    def backward_pass(self, iteration: int, loss_value: torch.Tensor):
        """Overrides UNet3DTrainer.backward_pass"""
        if self.flags.amp:
            self.scaler.scale(loss_value).backward()
        else:
            loss_value.backward()

        if (iteration + 1) % self.flags.ga_steps == 0:
            if self.flags.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
