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

import random
from enum import IntEnum
from einops import rearrange
import albumentations.augmentations.functional as F
import torch
import torchvision.io as io

from .functional import jpg_compression


# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# from functional import jpg_compression
# -

# https://github.com/albumentations-team/albumentations/blob/9b0525f479509195a7a7b7c19311d8e63bbc6494/albumentations/augmentations/transforms.py#L219

# + editable=true slideshow={"slide_type": ""}
class JPEGCompression(object):
    """Decreases image quality by Jpeg compression of an image.
    Args:
        quality_lower (float): lower bound on the image quality.
                               Should be in [0, 100] range for jpeg.
        quality_upper (float): upper bound on the image quality.
                               Should be in [0, 100] range for jpeg.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        quality_lower=60,
        quality_upper=100,
        consistent_quality=True,
        data_format="TCHW",
        p=0.5,
    ):
        super().__init__()

        assert 0 <= quality_lower <= quality_upper <= 100
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.consistent_quality = consistent_quality
        self.p = p
        self.debug = 0
        self.data_format = data_format

    def compress_img(self, x, quality=-1):
        assert x.ndim == 3 and x.shape[0] in [1, 3]
        if quality == -1:
            quality = random.randint(self.quality_lower, self.quality_upper)
        y = jpg_compression(x, quality)
        # print(quality, torch.sum(y-x))
        return y

    def __call__(self, x, **kwargs):
        """
        Args:
            x: (T, C, H, W) or (C, H, W)
        """
        # if not self.debug:
        #     print("Use jpeg compression augmentation")
        #     self.debug = True

        if random.random() > self.p:
            return x

        
        assert x.ndim in [3, 4]
        if self.consistent_quality:
            quality = random.randint(self.quality_lower, self.quality_upper)
        else:
            quality = -1

        if x.ndim == 3:
            return self.compress_img(x, quality)
        else:
            if self.data_format == "CTHW":
                x = rearrange(x, 'c t h w -> t c h w')
            x = torch.stack(
                [self.compress_img(x[i], quality) for i in range(x.shape[0])]
            )
            if self.data_format == "CTHW":
                x = rearrange(x, 't c h w -> c t h w')
            return x

# + tags=["active-ipynb"]
# x = torch.randint(0, 255, (10, 3, 224, 224)).to(torch.uint8)
#
# model = JPEGCompression(quality_lower=1, quality_upper=2)
#
# import time
#
# x = torch.randint(0, 255, (10, 3, 224, 224)).to(torch.uint8)
# s = time.time()
# y = model(x)
# e = time.time()
# print(e - s, torch.sum(y - x))
