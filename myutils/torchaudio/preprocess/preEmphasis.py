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

# +
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97) -> None:
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.ndim in [2, 3]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # reflect padding to match lengths of in/out
        x = F.pad(x, (1, 0), "reflect")
        return F.conv1d(x, self.flipped_filter)
