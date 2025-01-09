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

# 来自Dual-Level

# +
import torch
import torch.nn.functional as F


class AUCLoss(torch.nn.Module):
    def __init__(self, device, gamma=0.15, alpha=0.6, p=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.p = p
        self.device = device

    def forward(self, y_pred, y_true):
        y_true = y_true.reshape(-1, 1).float()
        
        pred = torch.softmax(y_pred, dim=-1)[:, 1:]
        pos = pred[torch.where(y_true == 0)]
        neg = pred[torch.where(y_true == 1)]
        pos = torch.unsqueeze(pos, 0)
        neg = torch.unsqueeze(neg, 1)
        diff = torch.zeros_like(pos * neg, device=self.device) + pos - neg - self.gamma
        masked = diff[torch.where(diff < 0.0)]
        auc = torch.mean(torch.pow(-masked, self.p))
        bce = F.binary_cross_entropy(pred, y_true)
        if masked.shape[0] == 0:
            loss = bce
        else:
            loss = self.alpha * bce + (1 - self.alpha) * auc
        return loss

# + tags=["active-ipynb"]
# y1 = torch.rand(8, 2)
# y2 = torch.rand(8, 1)
#
# loss = AUCLoss(device='cpu')
#
# loss(y1, y2)
