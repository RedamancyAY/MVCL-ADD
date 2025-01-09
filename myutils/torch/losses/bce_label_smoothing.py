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
import torch
import torch.nn.functional as F
import torch.nn as nn


# + tags=[]
class LabelSmoothingBCE(nn.Module):
    
    def __init__(self, label_smoothing):
        super().__init__()
        self.label_smoothing = label_smoothing
        print("BCE loss with label smoothing: ", self.label_smoothing)
        
    def forward(self, y_pred, y_true):
        '''
        Args:
            y_pred: (B, 1)
            y_true: (B,)
        '''
        assert y_true.ndim == 1
        if self.label_smoothing != 0:
            y_true = y_true.float() * (1 - self.label_smoothing) + self.label_smoothing / 2
        if y_pred.ndim == 2:
            y_pred = y_pred.squeeze(1)
        return F.binary_cross_entropy_with_logits(y_pred, y_true.float())

# + tags=["active-ipynb", "style-student"]
# x = torch.randn(1, 1)
# y = torch.randint(0, 2, (1,))
#
# module = LabelSmoothingBCE(label_smoothing=0.1)
# module(x, y)
