# ---
# jupyter:
#   jupytext:
#     formats: py:light
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
from myutils.tools import check_dir
# -

import copy
import json
from collections import defaultdict
from torchtnt.utils.flops import FlopTensorDispatchMode


class FLOPs_Callback(Callback):
    def __init__(self, batch_size_fn):
        super().__init__()
        self.has_test_flops = defaultdict(int)
        self.model = None
        self.batch_size_fn = batch_size_fn

    def common_batch_end(self, outputs, batch, stage="train"):

        if self.has_test_flops[stage]:
            return

        batch_size = self.batch_size_fn(batch)

        try:
            with FlopTensorDispatchMode(self.model) as ftdm:
                with torch.no_grad():
                    batch_res = self.model._shared_eval_step(batch, batch_idx=-1, stage=stage)
                flops_forward = copy.deepcopy(ftdm.flop_counts)
                self.has_test_flops[stage] = 1
            
            flops_forward_dict = dict(copy.deepcopy(flops_forward))
            flops_forward_dict['batch_size'] = batch_size
            flops_data = {
                "flops_forward": flops_forward_dict,
            }
            # Write the data to a JSON file
            with open(f"{self.log_dir}/flops_data_{stage}.json", "w") as json_file:
                json.dump(flops_data, json_file, indent=4)
            flops = sum(flops_forward[''].values()) / 1e9 / batch_size
        except Exception as e:

            print(r"Warning!!!!!!!, use FlopTensorDispatchMode but raise Exception, therefore, we try to use "
                 "lightning.fabric.utilities.throughput to compute FLOPs.")
            
            from lightning.fabric.utilities.throughput import measure_flops
            with torch.device("meta"):
                model = copy.deepcopy(self.model)
                x = copy.deepcopy(batch)
            model_fwd = lambda: model._shared_eval_step(x, batch_idx=-1, stage=stage)
            flops = measure_flops(model, model_fwd)
            flops = flops/2 / 1e9 / batch_size
        
        self.log_dict({f"flops_{stage}" : flops}, logger=True)
        self.has_test_flops[stage] = 1

    
    def common_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule, stage="train"
    ):
        self.model = pl_module
        self.log_dir = trainer.log_dir

    def common_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, stage="train"
    ):
        pass
        
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

