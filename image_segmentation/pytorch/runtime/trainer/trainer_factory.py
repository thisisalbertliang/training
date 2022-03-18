from runtime.trainer.cuda_trainer import CUDATrainer
from runtime.trainer.xla_trainer import XLATrainer


def get_trainer(flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks):
    """Initializes and return the trainer (XLATrainer or CUDATrainer)"""
    if flags.torch_xla:
        trainer = XLATrainer(
            flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks
        )
    else:
        trainer = CUDATrainer(
            flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks
        )
    return trainer
