from runtime.trainer.cuda_trainer import CUDATrainer
from runtime.trainer.xla_trainer import XLATrainer


def get_trainer(flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks):
    """Initializes and return the trainer (XLATrainer or CUDATrainer)"""
    if flags.device == "xla":
        trainer = XLATrainer(
            flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks
        )
    elif flags.device == "cuda":
        trainer = CUDATrainer(
            flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks
        )
    else:
        raise ValueError(f"Device {flags.device} unknown. Valid devices are: cuda, xla")
    return trainer
