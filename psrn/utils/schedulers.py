"""Learning rate schedulers for PSRN training."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    """Linear warmup scheduler.
    
    Used in Stage 2: warmup from 1e-6 to 1e-4 over 2k steps.
    From Sec. 3.5.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        start_lr: float,
        end_lr: float,
        last_epoch: int = -1,
    ) -> None:
        """Initialize warmup scheduler.
        
        Args:
            optimizer: optimizer to schedule
            warmup_steps: number of warmup steps
            start_lr: starting learning rate
            end_lr: target learning rate after warmup
            last_epoch: last epoch index
        """
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Get learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            progress = self.last_epoch / self.warmup_steps
            lr = self.start_lr + (self.end_lr - self.start_lr) * progress
            return [lr for _ in self.optimizer.param_groups]
        else:
            # After warmup, use base learning rate
            return [group["lr"] for group in self.optimizer.param_groups]

