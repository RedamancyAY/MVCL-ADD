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

# +
import os.path
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
# -

from myutils.torch.metrics import EER


class EER_Callback(Callback):
    def __init__(self, output_key, batch_key, theme="", num_classes=2):
        super().__init__()
        self.metrics = {}
        for stage in ["train", "val", "test", "pred"]:
            self.metrics[stage] = EER()
        self.output_key = output_key
        self.batch_key = batch_key
        self.theme = theme

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        for stage in self.metrics:
            self.reset_metric(stage)

    def reset_metric(self, stage):
        if not isinstance(self.metrics[stage], list):
            self.metrics[stage].reset()
        else:
            for metric in self.metrics[stage]:
                metric.reset()
    
    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        pass

    def common_batch_end(self, outputs, batch, stage="train", dataloader_idx=0):

        output = torch.nan_to_num(outputs[self.output_key].cpu())
        
        if dataloader_idx == 0:
            if isinstance(self.metrics[stage], list):
                metric = self.metrics[stage][0]
            else:
                metric = self.metrics[stage]
            metric.update(output, batch[self.batch_key].cpu())
        else:
            if not isinstance(self.metrics[stage], list):
                self.metrics[stage] = [self.metrics[stage], EER()]
            elif dataloader_idx >= len(self.metrics[stage]):
                self.metrics[stage].append(EER())
            metric = self.metrics[stage][dataloader_idx]
            metric.update(output, batch[self.batch_key].cpu())

    
    def common_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage="train",
        dataloader_idx=0,
        **kwargs,
    ):
        metric_name = "" if self.theme == "" else f"{self.theme}-"
        
        if not isinstance(self.metrics[stage], list):
            metric = self.metrics[stage]
            res = metric.compute()
            pl_module.log_dict(
                {f"{stage}-{metric_name}eer": res}, logger=True, prog_bar=True
            )
        else:
            for id, metric in enumerate(self.metrics[stage]):
                res = metric.compute()
                suffix = "" if id == 0 else "-dl%d"%(id)
                pl_module.log_dict(
                    {f"{stage}-{metric_name}eer{suffix}": res}, logger=True, prog_bar=True
                )
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.reset_metric('train')

    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset_metric('val')

    def on_test_epoch_start(self, trainer, pl_module):
        self.reset_metric('test')

    def on_predict_epoch_start(self, trainer, pl_module):
        self.reset_metric('pred')

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.common_batch_end(outputs=outputs, batch=batch, stage="pred")

    def on_predict_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.common_epoch_end(trainer=trainer, pl_module=pl_module, stage="pred")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.common_batch_end(outputs=outputs, batch=batch, stage="train")

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.common_epoch_end(trainer=trainer, pl_module=pl_module, stage="train")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.common_batch_end(outputs=outputs, batch=batch, stage="test")

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.common_epoch_end(trainer=trainer, pl_module=pl_module, stage="test")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx=0,
        **kwargs,
    ) -> None:
        self.common_batch_end(outputs=outputs, batch=batch, stage="val", dataloader_idx=dataloader_idx)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx=0, **kwargs
    ) -> None:
        self.common_epoch_end(trainer=trainer, pl_module=pl_module, stage="val", dataloader_idx=dataloader_idx)
