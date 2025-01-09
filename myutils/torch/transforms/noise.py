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

# + editable=true slideshow={"slide_type": ""}
# %load_ext autoreload
# %autoreload 2

# +
import random
from enum import IntEnum

import albumentations.augmentations.functional as F
import torch
import torchvision.io as io


# -

# 参考自：[class GaussNoise(ImageOnlyTransform):](https://github.com/albumentations-team/albumentations/blob/bcdf73d1a8da82a01cc25c67988200dc0b42f44c/albumentations/augmentations/transforms.py#L1293)

# + self.var_limit[0] editable=true slideshow={"slide_type": ""}
class GaussianNoise(object):
    """Apply gaussian noise to the input image.
    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        per_channel (bool): if set to True, noise will be sampled for each channel independently.
            Otherwise, the noise will be sampled once for all channels. Default: True
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, var_limit=(10.0, 50.0), mean=0.0, per_channel=True, p=0.5):
        super().__init__()
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0 or var_limit[1] < 0:
                raise ValueError("var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")
            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(
                    type(var_limit)
                )
            )

        self.mean = float(mean)
        self.per_channel = per_channel
        self.p = p

    def __call__(self, x, **kwargs):
        if random.random() <= self.p:
            return self.gaussian_noise(x)
        return x
        
    def gaussian_noise(self, x):
        assert x.ndim >= 3 and x.shape[-3] in [1, 3]
        dtype = x.dtype

        means = self.mean * torch.ones_like(x)

        if self.per_channel:
            sigma_shape = tuple(x.shape[:-2]) + (1, 1)
        else:
            sigma_shape = tuple(x.shape[:-3]) + (1, 1, 1)
        sigmas = (
            torch.rand(sigma_shape) * (self.var_limit[1] - self.var_limit[0])
        ) 

        # print(means, sigmas)
        gauss = torch.normal(means, sigmas)
        # print(gauss)
        x = x.to(torch.float32) + gauss
        x = torch.clip(x, min=0, max=255).to(dtype)
        return x

# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# N, C, H, W = 2, 3, 5, 5
# means = 0.0 * torch.ones(N, C, H, W)
# stds = torch.randint(10, 50, (N, C, 1, 1))
# torch.normal(means, stds).shape

# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# x = torch.randint(0, 255, (10, 3, 224, 224)).to(torch.uint8)
# model = GaussianNoise()
# import time
# s = time.time()
# y = model(x)
# e = time.time()
# print(y.shape, e - s)
# print('he')
