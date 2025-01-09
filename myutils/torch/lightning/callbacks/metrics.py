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

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
)


def color_print(*args):
    """
    print string with colorful background
    """
    from rich.console import Console
    string = ' '.join([str(x) for x in args])
    Console().print(f"[on #00ff00][#ff3300]{string}[/#ff3300][/on #00ff00]")


# # Base

# + editable=true slideshow={"slide_type": ""}
class Base(Callback):
    def __init__(
        self,
        output_key,
        batch_key,
        *args,
        theme="",
        avg_multi_dl_res=False,
        log_points=[],
        **kwargs,
    ):
        super().__init__()
        self.output_key = output_key
        self.batch_key = batch_key
        self.theme = theme

        # if val/test have multiple dataloaders, average the metrics among all dataloaders
        self.avg_multi_dl_res = avg_multi_dl_res
        self.avg_res = {}

        self.metrics = {}
        for stage in ["train", "val", "test", "pred"]:
            self.metrics[stage] = self.build_metric_funcs(*args, **kwargs)

        self.log_points = log_points

    @property
    def metric_name(self):
        raise NotImplementedError

    def build_metric_funcs(self, *args, **kwargs):
        raise NotImplementedError

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        for stage in self.metrics:
            self.reset_metric(stage)

    def reset_metric(self, stage):
        self.metrics[stage].reset()

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        pass

    def calculate_metric(self, metric_cls, step_outputs, batch_data):
        """

        For each sample, if its frame-wise labels have any zeros, we regrad it as 0;
        else, we regard it as 1.

        Args:
            metric_cls: the metric class to calculate acc
            step_outputs: a dict, we only need the `step_outputs[self.output_key]`, which is a tensor of shape (B, T)
            batch_data: a dict, we only need the batch_data[self.batch_key], which is a list of frame-wise labels
        """
        preds = step_outputs[self.output_key]
        targets = batch_data[self.batch_key]

        if metric_cls.device != preds.device:
            metric_cls = metric_cls.to(preds.device)
        metric_cls.update(preds, targets)

    def get_dataloader_number(self, trainer: Trainer, stage: str):
        stage_dls = {
            "val": trainer.val_dataloaders,
            "test": trainer.test_dataloaders,
            "pred": trainer.predict_dataloaders,
        }
        if stage == "train":
            return 1  # trainer only support training models with one train_dataloader
        else:
            if isinstance(stage_dls[stage], torch.utils.data.DataLoader):
                return 1
            if isinstance(stage_dls[stage], list):
                return len(stage_dls[stage])
            return 1

    def common_batch_end(
        self,
        outputs,
        batch,
        stage="train",
        dataloader_idx=0,
        trainer: Trainer = None,
        pl_module: LightningModule = None,
        *kwargs,
    ):
        if dataloader_idx == self.dataloader_idx:
            self.calculate_metric(self.metrics[stage], outputs, batch)
        else:
            self.common_epoch_end(trainer, pl_module, stage)
            self.reset_metric(stage)
            self.dataloader_idx = dataloader_idx
            self.calculate_metric(self.metrics[stage], outputs, batch)

        if not stage == "train":
            return
        if not self.log_points:
            return
        training_steps = trainer.num_training_batches
        global_steps = trainer.global_step
        cur_step = global_steps % training_steps
        log_steps = [int(training_steps * x) for x in self.log_points]
        if cur_step in log_steps:
            cur_epoch = (
                self.log_points[log_steps.index(cur_step)] + trainer.current_epoch
            )
            # print(log_steps, cur_step)
            theme = "" if self.theme == "" else f"{self.theme}-"
            monitor = f"{stage}-{theme}{self.metric_name}-middle_res"
            res = self.metrics[stage].compute()
            pl_module.logger.log_metrics({monitor: res}, step=cur_epoch)

    def common_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage="train",
        **kwargs,
    ):
        theme = "" if self.theme == "" else f"{self.theme}-"
        monitor = f"{stage}-{theme}{self.metric_name}"

        dataloader_num = self.get_dataloader_number(trainer, stage)
        if dataloader_num > 1:
            monitor = f"{monitor}-{self.dataloader_idx}"

        res = self.metrics[stage].compute()
        pl_module.log_dict(
            {monitor: res}, logger=True, prog_bar=True, add_dataloader_idx=False
        )

        # print(stage, monitor, res, trainer.logged_metrics, self.dataloader_idx)

        if dataloader_num == 1 or (not self.avg_multi_dl_res):
            return
        if self.dataloader_idx == 0:
            self.avg_res[stage] = []
        self.avg_res[stage].append(res)
        if self.dataloader_idx == dataloader_num - 1:
            monitor = monitor.replace(f"-{self.dataloader_idx}", "-avg")
            res = torch.mean(torch.stack(self.avg_res[stage]))
            pl_module.log_dict(
                {monitor: res}, logger=True, prog_bar=True, add_dataloader_idx=False
            )

        # print(stage, trainer.logged_metrics)

    def on_train_epoch_start(self, trainer, pl_module):
        self.reset_metric("train")
        self.dataloader_idx = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset_metric("val")
        self.dataloader_idx = 0

    def on_test_epoch_start(self, trainer, pl_module):
        self.reset_metric("test")
        self.dataloader_idx = 0

    def on_predict_epoch_start(self, trainer, pl_module):
        self.reset_metric("pred")
        self.dataloader_idx = 0

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.common_batch_end(
            outputs=outputs,
            batch=batch,
            stage="pred",
            trainer=trainer,
            pl_module=pl_module,
        )

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
        dataloader_idx=0,
        *args,
        **kwargs,
    ) -> None:
        self.common_batch_end(
            outputs=outputs,
            batch=batch,
            stage="train",
            trainer=trainer,
            pl_module=pl_module,
        )

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
        dataloader_idx=0,
        *args,
        **kwargs,
    ) -> None:
        self.common_batch_end(
            outputs=outputs,
            batch=batch,
            stage="test",
            trainer=trainer,
            pl_module=pl_module,
            dataloader_idx=dataloader_idx,
        )

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
        self.common_batch_end(
            outputs=outputs,
            batch=batch,
            stage="val",
            dataloader_idx=dataloader_idx,
            trainer=trainer,
            pl_module=pl_module,
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # print(self.metric_name, "Validation epoch end", self.val_dl_idx, self.x)
        self.common_epoch_end(trainer=trainer, pl_module=pl_module, stage="val")


# -

# # Video Level

class VideoFramewiseBinaryACC(Base):
    @property
    def metric_name(self):
        return "acc"

    def build_metric_funcs(self, *args, **kwargs):
        return BinaryAccuracy()

    def calculate_metric(self, metric_cls, step_outputs, batch_data):
        """
        The step_outputs[self.output_key] and batch_data[self.batch_key] is a tensor
        of shape (B, T), which denotes the logits of T frames.
        """
        # print(step_outputs[self.output_key].cpu(), batch_data[self.batch_key])

        # preds = step_outputs[self.output_key].cpu()
        preds = step_outputs[self.output_key]
        for i, _true in enumerate(batch_data[self.batch_key]):
            # true = _true.cpu()
            true = _true
            pred = preds[i, : true.shape[0]]
            # print(pred.shape, true.shape)
            # print("frame_level", pred, true)
            metric_cls.update(pred, true)


class VideoFramewiseBinaryAUC(VideoFramewiseBinaryACC):
    @property
    def metric_name(self):
        return "auc"

    def build_metric_funcs(self, *args, **kwargs):
        return BinaryAUROC()


# +
class VideoLevelBinaryACC(Base):
    @property
    def metric_name(self):
        return "acc"

    def build_metric_funcs(self, *args, **kwargs):
        return BinaryAccuracy()

    def calculate_metric(self, metric_cls, step_outputs, batch_data):
        """

        For each sample, if its frame-wise labels have any zeros, we regrad it as 0;
        else, we regard it as 1.

        Args:
            metric_cls: the metric class to calculate acc
            step_outputs: a dict, we only need the `step_outputs[self.output_key]`, which is a tensor of shape (B, T)
            batch_data: a dict, we only need the batch_data[self.batch_key], which is a list of frame-wise labels
        """
        # print(step_outputs[self.output_key].cpu(), batch_data[self.batch_key])

        # preds = step_outputs[self.output_key].cpu()
        preds = step_outputs[self.output_key]
        for i, _true in enumerate(batch_data[self.batch_key]):
            # true = _true.cpu()
            true = _true
            binary_true_video_label = (
                torch.tensor([1]) if torch.all(true > 0) else torch.tensor([0])
            ).to(true.device)
            pred = preds[i, : true.shape[0]]
            binary_pred_video_label = (
                torch.tensor([1]) if torch.all(pred > 0) else torch.tensor([0])
            ).to(true.device)
            # print(pred.shape, true.shape)
            # print("video_level", pred, binary_pred_video_label, true, binary_true_video_label)
            metric_cls.update(binary_pred_video_label, binary_true_video_label)


class VideoLevelBinaryAUC(VideoLevelBinaryACC):
    @property
    def metric_name(self):
        return "auc"

    def build_metric_funcs(self, *args, **kwargs):
        return BinaryAUROC()


# -

# # Image level

# ## Binary Classification

# +
class BinaryACC_Callback(Base):
    @property
    def metric_name(self):
        return "acc"

    def build_metric_funcs(self, *args, **kwargs):
        return BinaryAccuracy()

    
    def calculate_metric(self, metric_cls, step_outputs, batch_data):
        """

        For each sample, if its frame-wise labels have any zeros, we regrad it as 0;
        else, we regard it as 1.

        Args:
            metric_cls: the metric class to calculate acc
            step_outputs: a dict, we only need the `step_outputs[self.output_key]`, which is a tensor of shape (B, T)
            batch_data: a dict, we only need the batch_data[self.batch_key], which is a list of frame-wise labels
        
        """

        
        preds = step_outputs[self.output_key]
        if isinstance(batch_data, list):
            # print('!!!!!batch data is a list, use its index 1')
            batch_data = batch_data[1]
        targets = batch_data[self.batch_key]
        
        if metric_cls.device != preds.device:
            metric_cls = metric_cls.to(preds.device)
        metric_cls.update(preds, targets)


class BinaryAUC_Callback(BinaryACC_Callback):
    @property
    def metric_name(self):
        return "auc"

    def build_metric_funcs(self, *args, **kwargs):
        return BinaryAUROC()


# +
class _BinaryClsCount:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total, self.label_0, self.label_1 = 0, 0, 0
        self.label_0_right = 0
        self.label_1_right = 0

    def update(self, preds, targets):
        n = len(preds)
        self.total += n
        for i in range(n):
            # print(targets[i], preds[i], targets[i] == preds[i])
            if targets[i] == 0:
                self.label_0 += 1
                self.label_0_right += 1 if targets[i] == preds[i] else 0
            else:
                self.label_1 += 1
                self.label_1_right += 1 if targets[i] == preds[i] else 0

    def __call__(self, preds, targets):
        self.reset()
        self.update(preds, targets)
        return self.compute()
    
    def compute(self):
        acc = (self.label_0_right + self.label_1_right) / self.total
        acc0 = self.label_0_right / self.label_0 if self.label_0 > 0 else 1.0
        acc1 = self.label_1_right / self.label_1 if self.label_1 > 0 else 1.0

        print(
            f"Number: total/0/1 is {self.total}/{self.label_0}/{self.label_1}, "
            f"ACC on total/0/1 is {'%.4f'%acc}/{'%.4f'%acc0}/{'%.4f'%acc1}"
        )

        return acc0


class BinaryClsCount_Callback(Base):
    @property
    def metric_name(self):
        return "n_acc"

    def build_metric_funcs(self, *args, **kwargs):
        return _BinaryClsCount()

    def calculate_metric(self, metric_cls, step_outputs, batch_data):
        """

        For each sample, if its frame-wise labels have any zeros, we regrad it as 0;
        else, we regard it as 1.

        Args:
            metric_cls: the metric class to calculate acc
            step_outputs: a dict, we only need the `step_outputs[self.output_key]`, which is a tensor of shape (B, T)
            batch_data: a dict, we only need the batch_data[self.batch_key], which is a list of frame-wise labels
        """
        preds = step_outputs[self.output_key]
        preds = (torch.sigmoid(preds) + 0.50000001).int()
        targets = batch_data[self.batch_key]
        # if metric_cls.device != preds.device:
        # metric_cls = metric_cls.to(preds.device)
        metric_cls.update(preds, targets)


# -

# ## Multiclass Classification

# +
class MulticlassACC_Callback(Base):
    @property
    def metric_name(self):
        return "acc"

    def build_metric_funcs(self, num_classes, *args, **kwargs):
        return MulticlassAccuracy(num_classes=num_classes)

    def calculate_metric(self, metric_cls, step_outputs, batch_data):
        """

        For each sample, if its frame-wise labels have any zeros, we regrad it as 0;
        else, we regard it as 1.

        Args:
            metric_cls: the metric class to calculate acc
            step_outputs: a dict, we only need the `step_outputs[self.output_key]`, which is a tensor of shape (B, T)
            batch_data: a dict, we only need the batch_data[self.batch_key], which is a list of frame-wise labels
        """
        preds = step_outputs[self.output_key]
        targets = batch_data[self.batch_key]

        # preds = torch.nn.functional.softmax(preds, dim=1).max(dim=1)[1]
        # print(preds.shape, preds, targets.shape, targets)
        if metric_cls.device != preds.device:
            metric_cls = metric_cls.to(preds.device)
        metric_cls.update(preds, targets)


class MulticlassAUC_Callback(MulticlassACC_Callback):
    @property
    def metric_name(self):
        return "auc"

    def build_metric_funcs(self, num_classes, *args, **kwargs):
        return MulticlassAUROC(num_classes=num_classes)


# -

# ## Audio

# +
class AudioPESQ_Callback(Base):
    @property
    def metric_name(self):
        return "PESQ"

    def build_metric_funcs(self, sr=16000, *args, **kwargs):
        mode = "wb" if sr == 16000 else "nb"
        from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
        return PerceptualEvaluationSpeechQuality(fs=sr, mode=mode)

    def calculate_metric(self, metric_cls, step_outputs, batch_data):
        """

        For each sample, if its frame-wise labels have any zeros, we regrad it as 0;
        else, we regard it as 1.

        Args:
            metric_cls: the metric class to calculate acc
            step_outputs: a dict, we only need the `step_outputs[self.output_key]`, which is a tensor of shape (B, T)
            batch_data: a dict, we only need the batch_data[self.batch_key], which is a list of frame-wise labels
        """
        preds = step_outputs[self.output_key]
        targets = batch_data[self.batch_key]

        if metric_cls.device != preds.device:
            metric_cls = metric_cls.to(preds.device)
        metric_cls.update(preds, targets)
        

class AudioSNR_Callback(Base):
    @property
    def metric_name(self):
        return "SNR"

    def build_metric_funcs(self, *args, **kwargs):
        from torchmetrics.audio import SignalNoiseRatio
        return SignalNoiseRatio()
    
class AudioSDR_Callback(Base):
    @property
    def metric_name(self):
        return "SDR"

    def build_metric_funcs(self, *args, **kwargs):
        from torchmetrics.audio import SignalDistortionRatio
        return SignalDistortionRatio()

class AudioSISDR_Callback(Base):
    @property
    def metric_name(self):
        return "SI-SDR"

    def build_metric_funcs(self, *args, **kwargs):
        from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
        return ScaleInvariantSignalDistortionRatio()
        
class AudioSISNR_Callback(Base):
    @property
    def metric_name(self):
        return "SI-SNR"

    def build_metric_funcs(self, *args, **kwargs):
        from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
        return ScaleInvariantSignalNoiseRatio()
        
class AudioPSNR_Callback(Base):
    @property
    def metric_name(self):
        return "PSNR"

    def build_metric_funcs(self, *args, **kwargs):
        from torchmetrics.image import PeakSignalNoiseRatio
        return PeakSignalNoiseRatio()
