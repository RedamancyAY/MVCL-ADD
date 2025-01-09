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
import os
import statistics
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ay2.torch.deepfake_detection import DeepfakeAudioClassification
from ay2.torch.losses import (
    BinaryTokenContrastLoss,
    CLIPLoss1D,
    Focal_loss,
    LabelSmoothingBCE,
    MultiClass_ContrastLoss,
)
from ay2.torch.optim import Adam_GC
from ay2.torch.optim.selective_weight_decay import (
    Optimizers_with_selective_weight_decay,
    Optimizers_with_selective_weight_decay_for_modulelist,
)
from ay2.torchaudio.transforms import AddGaussianSNR
from ay2.torchaudio.transforms.self_operation import (
    AudioToTensor,
    CentralAudioClip,
    RandomAudioClip,
    RandomPitchShift,
    RandomSpeed,
)
from tqdm.auto import tqdm

# -

from ay2.tools import (
    find_unsame_name_for_file,
    freeze_modules,
    rich_bar,
    unfreeze_modules,
)

from ay2.torchaudio.transforms import SpecAugmentBatchTransform
from .utils import TimeFrequencyReconstructionLoss


# + editable=true slideshow={"slide_type": ""}
try:
    from .multiView_model import MultiViewModel
    from .utils import TimeFrequencyReconstructionLoss
except ImportError:
    from multiView_model import MultiViewModel
    from utils import TimeFrequencyReconstructionLoss


# + editable=true slideshow={"slide_type": ""}
class MultiViewModel_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        self.model = MultiViewModel(cfg=cfg, args=args)
        self.cfg = cfg
        self.args = args

        self.spec_transform = SpecAugmentBatchTransform.from_policy(cfg.aug_policy)

        dims = 512
        self.clip_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 768, bias=False),
                    # nn.BatchNorm1d(768),
                    nn.ReLU(True),
                    nn.Dropout(0.1),
                    nn.Linear(768, 768, bias=False),
                ),
                nn.Sequential(
                    nn.Linear(768, 512, bias=False),
                    # nn.BatchNorm1d(512),
                    nn.ReLU(True),
                    nn.Dropout(0.1),
                    nn.Linear(512, 512, bias=False),
                ),
            ]
        )

        self.stage1_epochs = 2

        self.configure_loss_fn()
        self.save_hyperparameters()

    def configure_loss_fn(
        self,
    ):
        self.bce_loss = LabelSmoothingBCE(label_smoothing=0.1)
        self.contrast_loss2 = BinaryTokenContrastLoss(alpha=0.1)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.clip_loss = CLIPLoss1D()
        self.reconstruction_loss = TimeFrequencyReconstructionLoss()

    def configure_optimizers(self):
        # optimizer = Optimizers_with_selective_weight_decay_for_modulelist(
        #     [self],
        #     optimizer="Adam",
        #     lr=0.0001,
        #     weight_decay=0.0001,
        # )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if "model.feature_model1D" in n
                    ],
                    "lr": 5e-5,
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if "model.feature_model2D" in n
                    ],
                    "lr": 5e-5,
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if "squeeze" in n or "expand" in n
                    ],
                    "lr": 5e-5,
                },
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if (not "model.feature_model1D" in n)
                        and (not "model.feature_model2D" in n) and (not "squeeze" in n) and (not "expand" in n)
                    ],
                    "lr": 1e-4,
                },
            ],
            weight_decay=1e-4,
        )
        return [optimizer]

    def get_shuffle_ids(self, B):
        shuffle_ids = torch.randperm(B)
        while 0 in (shuffle_ids - torch.arange(B)):
            shuffle_ids = torch.randperm(B)
        return shuffle_ids

    def calcuate_loss(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]
        label = batch["label"]
        losses = {}

        ### inter contrastive loss between feat1D and feat2D
        feat1D = batch_res["feature1D"]
        feat2D = batch_res["feature2D"]
        losses["clip_loss"] = self.clip_loss(
            feat1D, self.clip_heads[0](feat2D)
        ) + self.clip_loss(feat2D, self.clip_heads[1](feat1D))
        losses["mse_loss"] = self.reconstruction_loss(
            batch_res["raw_wav_feat"], batch_res["raw_spec"]
        )

        losses["cls_loss1D"] = self.bce_loss(batch_res["logit1D"], label)
        losses["cls_loss2D"] = self.bce_loss(batch_res["logit2D"], label)
        losses["cls_loss"] = self.bce_loss(batch_res["logit"], label)
        losses["contrast_loss"] = self.contrast_loss2(batch_res["feature"], label)
        losses["contrast_loss1D"] = self.contrast_loss2(batch_res["feature1D"], label)
        losses["contrast_loss2D"] = self.contrast_loss2(batch_res["feature2D"], label)

        w_inner_cl = 1 if self.cfg.use_inner_CL else 0
        w_inter_cl = 1 if self.cfg.use_inter_CL else 0
        w_cls_12 = 1 if self.cfg.use_cls_loss_1_2 else 0
        w_cls_mse = 1 if self.cfg.use_mse_loss else 0

        # if self.trainer.current_epoch < self.stage1_epochs:
        #     losses["loss"] = 0.3 * (losses["clip_loss"] + losses["contrast_loss1D"] + losses["contrast_loss2D"]) + 0.01 * losses["cls_loss"] + 0.01 * (losses["cls_loss1D"] + losses["cls_loss2D"])
        # else:
        #     losses["loss"] = 1.0 * losses["cls_loss"] + 0.1 * (losses["cls_loss1D"] + losses["cls_loss2D"]) + 0.01 * (losses["clip_loss"] + losses["contrast_loss1D"] + losses["contrast_loss2D"])
        # return losses

        losses["loss"] = (
            self.cfg.w_con
            * (
                w_inter_cl * losses["clip_loss"]
                + w_inner_cl * losses["contrast_loss1D"]
                + w_inner_cl * losses["contrast_loss2D"]
            )
            + self.cfg.w_cls * (losses["cls_loss"] + w_cls_12 * losses["cls_loss1D"] + w_cls_12 * losses["cls_loss2D"])
        )

        # losses["loss"] = (
        #     1.0 * losses["cls_loss"]
        #     # + 0.1 * (losses["clip_loss"] + losses['mse_loss'])
        #     + w_inter_cl * (losses["clip_loss"])
        #     + w_cls_mse * losses["mse_loss"]
        #     + w_inner_cl * (losses["contrast_loss1D"] + losses["contrast_loss2D"])
        #     + w_cls_12 * (losses["cls_loss1D"] + losses["cls_loss2D"])
        # )

        return losses

    def remove_parameters_from_total(self, total, removed):
        removed_ids = [id(x) for x in removed]
        new = []
        for x in total:
            if not id(x) in removed_ids:
                new.append(x)
        return new

    def _shared_pred(self, batch, batch_idx, stage="train"):
        """common predict step for train/val/test

        Note that the data augmenation is done in the self.model.feature_extractor.

        """
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        # if self.trainer.current_epoch < 2:
        #     indexs = (batch['label'] == 1)
        #     audio = audio[indexs]
        #     batch['label'] = batch['label'][indexs]

        batch_res = self.model(
            audio,
            stage=stage,
            batch=batch if stage == "train" else None,
            # spec_aug = self.spec_transform if stage == "train" else None,
            trainer=self.trainer,
            # freeze_feature_extractor=True if self.trainer.current_epoch >= self.stage1_epochs else False
        )

        batch_res["pred"] = (torch.sigmoid(batch_res["logit"]) + 0.5).int()

        if stage == "test" and self.args.log:
            with open("test-mutliview.txt", "a") as f:
                for i, (pred, label) in enumerate(
                    zip(batch_res["pred"], batch["label"])
                ):
                    # if torch.sign(batch_res["logit"][i]) != torch.sign(batch_res["logit1D"][i]) and torch.sign(batch_res["logit"][i]) != torch.sign(batch_res["logit2D"][i]):
                    if pred != label:
                        f.write(
                            " ".join(
                                str(a)
                                for a in [
                                    batch_idx,
                                    label.item(),
                                    batch_res["logit1D"][i].item(),
                                    batch_res["logit2D"][i].item(),
                                    batch_res["logit"][i].item(),
                                    batch["source"][i]
                                    if "source" in batch.keys()
                                    else "NoSource",
                                ]
                            )
                            + "\n"
                        )

        return batch_res
