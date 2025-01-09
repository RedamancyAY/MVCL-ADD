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

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .base import BinaryClassification


class DeepfakeAudioClassification(BinaryClassification):
    
    
    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        # print(audio.shape)
        
        # out = self.model(audio)

        feature = self.model.extract_feature(audio)
        out = self.model.make_prediction(feature)
        
        out = out.squeeze(-1)
        batch_pred = (torch.sigmoid(out) + 0.5).int()
        return {
            "logit": out,
            "pred": batch_pred,
            "feature": feature
        }
