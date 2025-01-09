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

import torch
import torch.nn as nn
import torchmetrics
import numpy as np

from .functional import calculate_eer, compute_eer


# + tags=["style-activity", "active-ipynb"]
# from functional import calculate_eer, compute_eer
# -

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
