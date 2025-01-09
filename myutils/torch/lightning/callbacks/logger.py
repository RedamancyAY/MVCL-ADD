# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

__all__ = ["LrLogger"]


class LrLogger(Callback):
    """Log learning rate in each epoch start."""

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        for i, optimizer in enumerate(trainer.optimizers):
            for j, params in enumerate(optimizer.param_groups):
                key = f"opt{i}_lr{j}"
                value = params["lr"]
                # pl_module.logger.log_metrics({key: value}, step=trainer.global_step)
                # pl_module.log(key, value, logger=False)
                pl_module.log_dict({key: value}, logger=True, prog_bar=True)
