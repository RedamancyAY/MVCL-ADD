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

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# Convert the function into an nn.Module using a lambda function
class LambdaFunctionModule(nn.Module):
    def __init__(self, torch_func):
        super().__init__()
        self.func = torch_func
        
    def forward(self, x):
        return self.func(x)
