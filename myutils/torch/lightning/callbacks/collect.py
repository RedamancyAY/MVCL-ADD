# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=[]
# %load_ext autoreload
# %autoreload 2

# + tags=[]
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
from myutils.tools import check_dir


# -

class Collect_Callback(Callback):
    def __init__(self, output_keys, batch_keys, save_path):
        super().__init__()
        self.output_keys = output_keys
        self.batch_keys = batch_keys
        self.save_path = save_path
        self.res = {"train": {}, "val": {}, "test": {}, "pred": {}}

    def common_batch_end(self, outputs, batch, stage="train"):
        # print(outputs.keys(), batch.keys(), self.res[stage].keys())
        for key in self.output_keys:
            if key in outputs.keys():
                self.res[stage][key].append(outputs[key])
        for key in self.batch_keys:
            if key in batch.keys():
                self.res[stage][key].append(batch[key])

    def common_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule, stage="train"
    ):
        self.res[stage].clear()
        for key in self.output_keys + self.batch_keys:
            self.res[stage][key] = []

    def common_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, stage="train"
    ):
        res = self.res[stage]
        for key in res:
            if res[key]:
                res[key] = torch.concat(res[key], dim=0).detach().cpu().numpy()
                print(key, res[key].shape)
        save_path = os.path.join(self.save_path, stage + '.npz')
        check_dir(save_path)
        from myutils.tools import find_unsame_name_for_file
        save_path = find_unsame_name_for_file(save_path)
        np.savez(save_path, **res)
        print(f"Collect_Callback: Save collection res in {stage} stage at {save_path}.")
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.common_epoch_start(trainer=trainer, pl_module=pl_module, stage="test")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.common_epoch_start(trainer=trainer, pl_module=pl_module, stage="val")

    def on_test_epoch_start(self, trainer, pl_module):
        self.common_epoch_start(trainer=trainer, pl_module=pl_module, stage="test")

    def on_predict_epoch_start(self, trainer, pl_module):
        self.common_epoch_start(trainer=trainer, pl_module=pl_module, stage="pred")

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
    ) -> None:
        self.common_batch_end(outputs=outputs, batch=batch, stage="val")

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self.common_epoch_end(trainer=trainer, pl_module=pl_module, stage="val")
