# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import re

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback


# -

class EarlyStoppingWithMinimumEpochs(pl.callbacks.EarlyStopping):
    """
    Early stopping model after training with minimum epochs. In the first minimum epochs,
    this callback will not run early_stopping_check.

    Args:
        min_epochs: the minimum epochs for tarining.
    """

    def __init__(self, min_epochs=3, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = min_epochs

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch < self.min_epochs:
            return

        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch < self.min_epochs:
            return

        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)


class EarlyStoppingWithLambdaMonitor(pl.callbacks.EarlyStopping):
    """
    Early stopping model after training with minimum epochs. In the first minimum epochs,
    this callback will not run early_stopping_check.

    Args:
        min_epochs: the minimum epochs for tarining.
    """

    def _validate_condition_metric(self, logs):
        if "+++" in self.monitor:
            monitor1 = self.monitor.split("+++")[0].strip()
            monitor2 = self.monitor.split("+++")[1].strip()

        v1 = logs.get(monitor1)
        v2 = logs.get(monitor2)
        logs[self.monitor] = (v1 + v2) / 2

        if v1 is None or v2 is None:
            raise RuntimeError(error_msg)
        return super()._validate_condition_metric(logs)

    def log_monitor(self, trainer, pl_module):
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.callback_metrics
        pl_module.log_dict(
            {self.monitor: logs[self.monitor]},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)
        self.log_monitor(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer)
        self.log_monitor(trainer, pl_module)



class EarlyStoppingLR(Callback):
    """Early stop model training when the LR is lower than threshold."""

    def __init__(self, lr_threshold: float, mode="all"):
        self.lr_threshold = lr_threshold

        if mode in ("any", "all"):
            self.mode = mode
        else:
            raise ValueError(f"mode must be one of ('any', 'all')")

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._run_early_stop_checking(trainer)

    def _run_early_stop_checking(self, trainer: pl.Trainer) -> None:
        metrics = trainer._logger_connector.callback_metrics
        if len(metrics) == 0:
            return
        all_lr = []
        for key, value in metrics.items():
            if re.match(r"opt\d+_lr\d+", key):
                all_lr.append(value)

        if len(all_lr) == 0:
            return

        if self.mode == "all":
            if all(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True
        elif self.mode == "any":
            if any(lr <= self.lr_threshold for lr in all_lr):
                trainer.should_stop = True

# + tags=["active-ipynb", "style-student"]
# s = EarlyStop(min_epochs=10, monitor="psnr")
