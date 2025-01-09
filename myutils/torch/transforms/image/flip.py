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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from typing import Union


# https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py

# ```python
# def hflip(img: Tensor) -> Tensor:
#     _assert_image_tensor(img)
#
#     return img.flip(-1)
# ```

class BatchRandomHorizontalFlip(nn.Module):
    '''
    Args:
        kernel_size: the kernel size of the 2d gaussian kernel
        sigma: [min_sigma, max_sigma], when creating new kernels, the module will
               randomly select a sigma from [min_sigma, max_sigma].
    '''
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    
    def forward(self, x, label=None):
        B = x.shape[0]
        p = torch.rand(B)
        index1 = torch.where(p <= self.p)
        if len(index1[0]) == 0:
            return x
        else:
            index2 = torch.where(p > self.p)
            x_flipped = x[index1].flip(-1)
            # x = torch.concat([x_flipped, x[index2]], dim=0)
            x[index1] = x_flipped
            return x.contiguous()

# + tags=["active-ipynb"]
# blur = BatchRandomHorizontalFlip(p=0.5)
# x = torch.rand(32, 30, 224, 224)
# y = torch.rand(32, 3, 10, 224, 224)
# print(blur(x).shape, blur(y).shape)

# + tags=["active-ipynb"]
# from io import BytesIO
#
# import numpy as np
# import requests as req
# from PIL import Image
# from myutils.visualization import Plot
#
# img_src = "https://img0.baidu.com/it/u=259444276,4281031064&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=313"
# response = req.get(img_src)
# image = Image.open(BytesIO(response.content))
# image = np.array(image)[:, :, ::-1]  # BGR -> RGB
#
# img = torch.Tensor(image.copy())
# video = torch.stack([img, img, img], dim=0)
# videos = torch.stack([video, video, video, video, video], dim=0)
# videos = rearrange(videos, 'b t h w c -> b c t h w')
#
# blur = BatchRandomHorizontalFlip(p=0.5)
#
# videos2 = blur(videos)
# Plot.plot_videos(videos2.numpy())
