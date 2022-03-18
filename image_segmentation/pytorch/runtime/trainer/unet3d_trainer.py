import time
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Iterator

import torch
from model.unet3d import Unet3D
from runtime.distributed.distributed_utils import get_world_size, reduce_tensor
from runtime.inference import evaluate
from runtime.logging import CONSTANTS, mllog_end, mllog_event, mllog_start
from torch.nn import Parameter
from torch.optim import SGD, Adam


class UNet3DTrainer(ABC):
    """Base class for training UNet3D in PyTorch CUDA and PyTorch/XLA"""

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
        super().__init__()
        # Save training arguments
        self.flags = flags
        self.callbacks = callbacks
        self.score_fn = score_fn
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model: Unet3D = model
        self.loss_fn: torch.nn.Module = loss_fn

        # move model and loss_fn to device
        self.model.to(device)
        self.loss_fn.to(device)
        # setup optimizer
        self.optimizer = UNet3DTrainer.get_optimizer(self.model.parameters(), flags)
        if flags.lr_decay_epochs:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=flags.lr_decay_epochs,
                gamma=flags.lr_decay_factor,
            )

    def train(self):
        """Trains the UNet3D model"""
        is_successful = False
        diverged = False
        next_eval_at = self.flags.start_eval_at
        world_size = get_world_size()
        is_distributed = world_size > 1

        self.model.train()

        for callback in self.callbacks:
            callback.on_fit_start()

        for epoch in range(1, self.flags.epochs + 1):
            cumulative_loss = []

            # learning rate warm up
            if 0 < epoch <= self.flags.lr_warmup_epochs:
                UNet3DTrainer.lr_warmup(
                    self.optimizer,
                    self.flags.init_learning_rate,
                    self.flags.learning_rate,
                    epoch,
                    self.flags.lr_warmup_epochs,
                )

            mllog_start(
                key=CONSTANTS.BLOCK_START,
                sync=False,
                metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch, CONSTANTS.EPOCH_COUNT: 1},
            )
            mllog_start(
                key=CONSTANTS.EPOCH_START,
                metadata={CONSTANTS.EPOCH_NUM: epoch},
                sync=False,
            )

            if is_distributed:
                # shuffle the train data for the current epoch
                self.train_sampler.set_epoch(epoch)

            loss_value = None
            for iteration, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                image, label = batch
                image, label = image.to(self.device), label.to(self.device)

                for callback in self.callbacks:
                    callback.on_batch_start()

                loss_value = self.forward_pass(image, label)
                self.backward_pass(iteration, loss_value)

                loss_value = reduce_tensor(loss_value).detach().cpu().numpy()
                cumulative_loss.append(loss_value)
                # in debug mode, log the train loss on each iteration
                if self.flags.debug:
                    mllog_event(
                        key="train_loss",
                        value=loss_value,
                        metadata={
                            CONSTANTS.EPOCH_NUM: epoch,
                            "iteration_num": iteration,
                        },
                        sync=False,
                    )

            mllog_end(
                key=CONSTANTS.EPOCH_STOP,
                metadata={
                    CONSTANTS.EPOCH_NUM: epoch,
                    "current_lr": self.optimizer.param_groups[0]["lr"],
                },
                sync=False,
            )

            # startup time is defined as the time between the program
            # starting and the 1st epoch ending
            if epoch == 1:
                mllog_end(
                    key="startup_time",
                    value=time.time() - self.flags.program_start_time,
                    metadata={CONSTANTS.EPOCH_NUM: epoch},
                )

            if self.flags.lr_decay_epochs:
                self.scheduler.step()

            if epoch == next_eval_at:
                next_eval_at += self.flags.evaluate_every
                mllog_start(
                    key=CONSTANTS.EVAL_START,
                    value=epoch,
                    metadata={CONSTANTS.EPOCH_NUM: epoch},
                    sync=False,
                )

                # run model evaluation on the validation data
                eval_metrics = evaluate(
                    self.flags,
                    self.model,
                    self.val_loader,
                    self.loss_fn,
                    self.score_fn,
                    self.device,
                    epoch,
                )
                eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)

                mllog_event(
                    key=CONSTANTS.EVAL_ACCURACY,
                    value=eval_metrics["mean_dice"],
                    metadata={CONSTANTS.EPOCH_NUM: epoch},
                    sync=False,
                )
                mllog_end(
                    key=CONSTANTS.EVAL_STOP,
                    metadata={CONSTANTS.EPOCH_NUM: epoch},
                    sync=False,
                )

                for callback in self.callbacks:
                    callback.on_epoch_end(
                        epoch=epoch,
                        metrics=eval_metrics,
                        model=self.model,
                        optimizer=self.optimizer,
                    )
                self.model.train()
                if eval_metrics["mean_dice"] >= self.flags.quality_threshold:
                    is_successful = True
                elif eval_metrics["mean_dice"] < 1e-6:
                    print("MODEL DIVERGED. ABORTING.")
                    diverged = True

            mllog_end(
                key=CONSTANTS.BLOCK_STOP,
                metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch, CONSTANTS.EPOCH_COUNT: 1},
                sync=False,
            )

            if is_successful or diverged:
                break

    @abstractmethod
    def forward_pass(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Runs the model forward pass on the image and label

        :param torch.Tensor image: the input image from train data
        :param torch.Tensor label: the label associated with the input image from train data
        :return: the loss value
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def backward_pass(self, iteration: int, loss_value: torch.Tensor):
        """
        Runs the model backward pass, given the loss value from the forward pass

        :param int iteration: the iteration number in the current epoch
        :param torch.Tensor loss_value: the loss value from the forward pass
        """
        pass

    @staticmethod
    def lr_warmup(
        optimizer: torch.optim.Optimizer,
        init_lr: int,
        lr: int,
        current_epoch: int,
        warmup_epochs: int,
    ):
        """
        Warms up learning rate for the current epoch

        :param torch.optim.Optimizer optimizer: the optimizer
        :param int init_lr: the initial learning rate
        :param int lr: the current learning rate
        :param int current_epoch: the current epoch
        :param int warmup_epochs: number of epochs to warm up
        """
        scale = current_epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = init_lr + (lr - init_lr) * scale

    @staticmethod
    def get_optimizer(params: Iterator[Parameter], flags: Namespace) -> torch.optim.Optimizer:
        """
        Initializes the optimizer with the model weights

        :param Iterator[Parameter] params: the model weights
        :param Namespace flags: the runtime arguments
        :return: the initialized optimizer
        :rtype: torch.optim.Optimizer
        """
        if flags.optimizer == "adam":
            optim = Adam(params, lr=flags.learning_rate, weight_decay=flags.weight_decay)
        elif flags.optimizer == "sgd":
            optim = SGD(
                params,
                lr=flags.learning_rate,
                momentum=flags.momentum,
                nesterov=True,
                weight_decay=flags.weight_decay,
            )
        elif flags.optimizer == "lamb":
            import apex

            optim = apex.optimizers.FusedLAMB(
                params,
                lr=flags.learning_rate,
                betas=flags.lamb_betas,
                weight_decay=flags.weight_decay,
            )
        else:
            raise ValueError("Optimizer {} unknown.".format(flags.optimizer))
        return optim
