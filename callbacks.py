# +
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchmetrics
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, roc_curve
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

# -


color_progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="#008000",
        progress_bar="#0000ff",
        progress_bar_finished="#00ff00",
        progress_bar_pulse="red",
        batch_progress="bold #00ff00 on #ff0000",
        time="#ff0000",
        processing_speed="#0000ff",
        metrics="bold #ff0000 on #00ff00",
    )
)


def Color_progress_bar():
    return RichProgressBar(
        theme=RichProgressBarTheme(
            description="#008000",
            progress_bar="#0000ff",
            progress_bar_finished="#00ff00",
            progress_bar_pulse="red",
            batch_progress="bold #00ff00 on #ff0000",
            time="#ff0000",
            processing_speed="#0000ff",
            metrics="bold #ff0000 on #00ff00",
        ),
        leave=True,
    )


# +
def calculate_eer(y, y_score) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Args:
        y: the ture label for prediction
        y_socre: the logits for prediction
    Return:
        thresh, eer, fpr, tpr
    """

    fpr, tpr, thresholds = roc_curve(y, -y_score)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    # return thresh, eer, fpr, tpr
    return 1 - eer


"""
Python compute equal error rate (eer)
ONLY tested on binary classification

:param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
:param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
:param positive_label: the class that is viewed as positive class when computing EER
:return: equal error rate (EER)
"""


def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


class EER(torchmetrics.Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.y_true = []
        self.y_pred = []

    def calc_eer(self):
        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)

        if all(y_true == 0):
            return 1.0

        try:
            res = calculate_eer(y_true, y_pred)
        except ValueError as e:
            res = -1.0
            print("Error computing EER, return -1")
        return res

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.retains_grad:
            preds = preds.detach()
            target = preds.detach()
        if not preds.is_cpu:
            preds = preds.cpu()
            target = target.cpu()
        if isinstance(preds, torch.Tensor):
            preds = preds.numpy()
            target = target.numpy()

        self.y_pred.append(preds)
        self.y_true.append(target)

        # print(self.y_pred, self.y_true)

        # eer = self.calc_eer()
        # self.total = torch.tensor(eer)

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def compute(self):
        eer = self.calc_eer()
        self.total = torch.tensor(eer)
        return self.total


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
                suffix = "" if id == 0 else "-dl%d" % (id)
                pl_module.log_dict(
                    {f"{stage}-{metric_name}eer{suffix}": res},
                    logger=True,
                    prog_bar=True,
                )

    def on_train_epoch_start(self, trainer, pl_module):
        self.reset_metric("train")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset_metric("val")

    def on_test_epoch_start(self, trainer, pl_module):
        self.reset_metric("test")

    def on_predict_epoch_start(self, trainer, pl_module):
        self.reset_metric("pred")

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
        self.common_batch_end(
            outputs=outputs, batch=batch, stage="val", dataloader_idx=dataloader_idx
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx=0, **kwargs
    ) -> None:
        self.common_epoch_end(
            trainer=trainer,
            pl_module=pl_module,
            stage="val",
            dataloader_idx=dataloader_idx,
        )


def color_print(*args):
    """
    print string with colorful background
    """
    from rich.console import Console

    string = " ".join([str(x) for x in args])
    Console().print(f"[on #00ff00][#ff3300]{string}[/#ff3300][/on #00ff00]")


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


########################################


def common_callbacks():
    callbacks = [
        Color_progress_bar(),
        BinaryACC_Callback(batch_key="label", output_key="logit"),
        BinaryAUC_Callback(batch_key="label", output_key="logit"),
        EER_Callback(batch_key="label", output_key="logit"),
    ]

    return callbacks


def custom_callbacks(args, cfg):
    callbacks = []
    return callbacks


def training_callbacks(args):
    monitor = "val-auc"
    es = EarlyStopping

    callbacks = [
        # save last ckpt
        ModelCheckpoint(
            dirpath=None, save_top_k=0, save_last=True, save_weights_only=False
        ),
        # save best ckpt
        ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            monitor=monitor,
            mode="max",
            save_last=False,
            filename="best-{epoch}-{val-auc:.4f}",
            save_weights_only=True,
            verbose=True,
        ),
    ]

    if args.earlystop:
        callbacks.append(
            es(
                monitor=monitor,
                min_delta=0.001,
                patience=args.earlystop if args.earlystop > 1 else 3,
                mode="max",
                stopping_threshold=0.999,
                verbose=True,
            )
        )
    return callbacks


def make_callbacks(args, cfg):
    callbacks = common_callbacks()
    callbacks += custom_callbacks(args, cfg)
    if not args.test:
        callbacks += training_callbacks(args)

    return callbacks
