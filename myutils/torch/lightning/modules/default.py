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

# %load_ext autoreload
# %autoreload 2

import pytorch_lightning as pl
import torch
import torch.nn as nn


class DefaultLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def calcuate_loss(self, batch_res, batch, stage="train"):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def _shared_pred(self, batch, batch_idx, stage="train"):
        raise NotImplementedError

    def log_losses(self, losses, stage='train'):
        self.log_dict(
            {f"{stage}-{key}-step": losses[key] for key in losses},
            on_step=True,
            on_epoch=False,
            logger=False,
            prog_bar=True,
        )
        self.log_dict(
            {f"{stage}-{key}": losses[key] for key in losses},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

    def _shared_eval_step(self, batch, batch_idx, stage="train"):
        batch_res = self._shared_pred(batch, batch_idx, stage=stage)
        losses = self.calcuate_loss(batch_res, batch, stage=stage)
        self.log_losses(losses, stage=stage)
        batch_res.update(losses)
        return batch_res

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, batch_idx, stage="test")

    def prediction_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, batch_idx, stage="predict")
