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

# +
import random

import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange


# -

class RandomCropVideo(T.RandomCrop):

    """
    Inherited from torchvision.transforms.RandomCrop. It randomly crops the input video, where
    video is with shape of (T, H, W, C) or (T, C, H, W)
    """

    def __init__(self, input_shape="THWC", size=224, **kwargs):
        """
        Args:
            input_shape: 'THWC' or 'TCHW'
            size: the cropped size for the spatial dimension
        """
        super().__init__(size=size, **kwargs)
        self.input_shape = input_shape
        self.func_map = {"THWC": self.THWC, "TCHW": self.TCHW}

    def THWC(self, x):
        """
        crop video with shape of (T, H, W, C)
        """
        t, h, w, c = x.shape
        v = rearrange(x, "t h w c -> (t c) h w")
        v = super().forward(v)
        v = rearrange(v, "(t c) h w -> t h w c", t=t, c=c)
        return v

    def TCHW(self, x):
        """
        crop video with shape of (T, C, H, W)
        """
        x = super().forward(x)
        return x

    def __call__(self, x, *args):
        '''
        use the `super().forward` as default if the input_shape is not in `self.func_map.keys()`
        '''
        print(x.shape)
        if not self.input_shape in self.func_map.keys():
            return super().forward(x)
        return self.func_map[self.input_shape](x)


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# # Input video is with shape of (T, H, W, C)
# video = torch.randn(2, 254, 242, 3)
# t = RandomCropVideo(size=224, input_shape="THWC")
# print(t(video).shape)

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# # Input video is with shape of (T, C, H, W)
# video = torch.randn(2, 3, 254, 242)
# t = RandomCropVideo(size=224, input_shape="TCHW")
# print(t(video).shape)
# -

class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), p=1):
        self.transform = T.RandomResizedCrop((224, 224), scale=scale, ratio=ratio)
        self.p = p

    def deal5D(self, x):
        c, t = x.shape[1:3]
        x = rearrange(x, "b c t h w -> b (c t) h w")
        x = self.transform(x)
        x = rearrange(x, "b (c t) h w -> b c t h w", c=c, t=t)
        return x

    def __call__(self, x, *args):
        if random.random() > self.p:
            return x
        if len(x.shape) == 5:
            x = self.deal5D(x)
        else:
            x = self.transform(x)
        return x

# # 测试

# ## 4D
#

# + tags=["active-ipynb"]
# import torchvision
#
# from myutils.visualization import Plot
#
# model = RandomResizedCrop(size=224, scale=(0.5, 0.7))
# x, _, metadata = torchvision.io.read_video(
#     "/usr/local/ay_data/dataset/0-deepfake/DeepfakeTIMIT/lower_quality/fadg0/sa1-video-fram1.avi"
# )
# x = x[::24, ...]
# x = x.permute(0, 3, 1, 2)
# print(x.shape)
# y = model(x)
# Plot.plot_video(x)
# Plot.plot_video(y)
# -

# ## 5D

# + tags=["active-ipynb"]
# x = x[None, ...]
# y = model(x)
#
# Plot.plot_video(y[0])
