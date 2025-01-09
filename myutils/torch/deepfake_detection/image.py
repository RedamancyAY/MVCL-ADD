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

from .base import BinaryClassification


class DeepfakeImageClassification(BinaryClassification):
    
        
    def _shared_pred(self, batch, batch_idx):
        x = self.normalize_input(batch['img'])
        if hasattr(self.model, "detect_deepfake_img"):
            batch_out = self.model.detect_deepfake_img(x)
        else:
            batch_out = self.model(x)

        if batch_out.ndim == 2 and batch_out.shape[-1] == 1:
            batch_out = batch_out[:, 0]
        
        if batch_out.shape[-1] == 2:
            binary_logit = torch.softmax(batch_out, dim=-1)[:, 1]
            batch_pred = torch.argmax(torch.softmax(batch_out, dim=-1), dim=1).int()
        else:
            binary_logit = batch_out
            batch_pred = (torch.sigmoid(batch_out) + 0.5).int()

        
        return {"logit": batch_out, "pred": batch_pred, "binary_logit":binary_logit}
