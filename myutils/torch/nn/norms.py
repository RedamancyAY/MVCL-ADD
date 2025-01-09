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
import torch.nn as nn
from einops import rearrange


# + tags=[]
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, x):
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b h w c")
            x = self._norm(x)
            x = rearrange(x, "b h w c -> b c h w")
        elif x.ndim == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = self._norm(x)
            x = rearrange(x, "b t h w c -> b c t h w")
        elif x.ndim == 3:
            x = rearrange(x, "b c l -> b l c")
            x = self._norm(x)
            x = rearrange(x, "b l c -> b c l")
        return x
