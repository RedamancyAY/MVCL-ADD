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
# %load_ext autoreload
# %autoreload 2

# + tags=[]
import numbers
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# + tags=[]
def rgb2hls(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hls_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hls_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hls_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hls_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hls_h[cmax_idx == 3] = 0.0
    hls_h /= 6.0

    hls_l = (cmax + cmin) / 2.0
    hls_s = torch.empty_like(hls_h)
    hls_s[hls_l == 0] = 0
    hls_s[hls_l == 1] = 0
    hls_l_ma = torch.bitwise_and(hls_l > 0, hls_l < 1)
    hls_l_s0_5 = torch.bitwise_and(hls_l_ma, hls_l <= 0.5)
    hls_l_l0_5 = torch.bitwise_and(hls_l_ma, hls_l > 0.5)
    hls_s[hls_l_s0_5] = ((cmax - cmin) / (hls_l * 2.0))[hls_l_s0_5]
    hls_s[hls_l_l0_5] = ((cmax - cmin) / (-hls_l * 2.0 + 2.0))[hls_l_l0_5]
    return torch.cat([hls_h * 360, hls_l, hls_s], dim=1)


def hls2rgb(hls: torch.Tensor) -> torch.Tensor:
    hls_h, hls_l, hls_s = hls[:, 0:1] / 360, hls[:, 1:2], hls[:, 2:3]
    _c = (-torch.abs(hls_l * 2.0 - 1.0) + 1) * hls_s
    _x = _c * (-torch.abs(hls_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hls_l - _c / 2.0
    idx = (hls_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hls)
    _o = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


# + tags=["style-activity", "active-ipynb"]
# # 1. torch rgb2hls
# x = torch.rand(1, 3, 224, 224)
# y = rgb2hls(x)
# print(torch.mean(hls2rgb(y) - x))
#
#
# a = rearrange(x, "b c h w -> b h w c")
# b_hls = cv2.cvtColor(a[0].numpy(), cv2.COLOR_RGB2HLS)
# b = rearrange(b_hls[None, ...], "b h w c -> b c h w")
# b = torch.Tensor(b)
#
# z = b - y
#
# print(torch.mean(z))

# + tags=[]
class BatchISONoise(nn.Module):
    """
    Args:
        kernel_size: the kernel size of the 2d gaussian kernel
        sigma: [min_sigma, max_sigma], when creating new kernels, the module will
               randomly select a sigma from [min_sigma, max_sigma].
    """

    def __init__(self, color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5):
        super().__init__()
        self.color_shift = color_shift
        self.intensity = intensity
        self.p = p

    def _get_noise(self, N, H, W, std, T=None):
        color_shift = torch.empty(N).uniform_(self.color_shift[0], self.color_shift[1]).to(std.device)
        intensity = torch.empty(N).uniform_(self.intensity[0], self.intensity[1]).to(std.device)
        # print(intensity.device, std.device)
        if T is not None:
            color_shift = torch.repeat_interleave(color_shift, T)
            intensity = torch.repeat_interleave(intensity, T)
            N = N * T
        ones = torch.ones(N, H, W).to(std.device)
        luminance_noise = torch.poisson(ones * (std * intensity * 255)[..., None, None])
        color_noise = torch.normal(
            ones * 0, (color_shift * 360 * intensity)[..., None, None]
        )
        return luminance_noise, color_noise

    def _apply_noise(self, luminance_noise, color_noise, hls_img):
        h, l, s = hls_img[:, 0, ...], hls_img[:, 1, ...], hls_img[:, 2, ...]
        h += color_noise
        h[h < 0] += 360
        h[h > 360] -= 360
        l += (luminance_noise / 255) * (1.0 - l)
        new_hls = torch.stack([h, l, s], dim=1)
        new_img = hls2rgb(new_hls)
        return new_img
    
    def _iso_noise_4d(self, x, T=None):
        B, C, H, W = x.shape
        hls = rgb2hls(x)
        stddev = torch.std(hls, dim=[2, 3])
        # print("std is ", stddev.shape)

        luminance_noise, color_noise = self._get_noise(B, H, W, stddev[:, 1])
        new_img = self._apply_noise(luminance_noise, color_noise, hls)
        return new_img

    def noise_4D(self, x):
        B, C, H, W = x.shape
        if B < 1:
            return x
        
        p = torch.rand(B)
        index1 = torch.where(p <= self.p)
        index2 = torch.where(p > self.p)
        x_noise = self._iso_noise_4d(x[index1])
        # x = torch.concat([x_noise, x[index2]], dim=0)
        x[index1] = x_noise
        return x.contiguous()
    

    def _iso_noise_5d(self, x, T=None):
        B, C, T, H, W = x.shape
        
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        hls = rgb2hls(x)
        stddev = torch.std(hls, dim=[2, 3])
        # print("std is ", stddev.shape)
        luminance_noise, color_noise = self._get_noise(B, H, W, stddev[:, 1], T=T)
        new_img = self._apply_noise(luminance_noise, color_noise, hls)
        new_img = rearrange(new_img, '(b t) c h w -> b c t h w', b=B, t=T)
        return new_img
    
    
    def noise_5D(self, x):
        B, C, T, H, W = x.shape
        if B < 1:
            return x
        
        p = torch.rand(B)
        index1 = torch.where(p <= self.p)
        index2 = torch.where(p > self.p)
        x_noise = self._iso_noise_5d(x[index1])
        # x = torch.concat([x_noise, x[index2]], dim=0)
        x[index1] = x_noise
        return x.contiguous()

    def forward(self, x, label=None):
        if x.ndim == 4:
            return self.noise_4D(x)
        elif x.ndim == 5:
            return self.noise_5D(x)
        return x

# + tags=["active-ipynb"]
# blur = BatchISONoise()
# x = torch.rand(32, 3, 224, 224)
# y = torch.rand(32, 3, 10, 224, 224)
# print(blur(x).shape, blur(y).shape)
