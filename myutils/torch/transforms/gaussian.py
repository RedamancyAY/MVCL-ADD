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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from typing import Union


class BatchGaussianBlur(nn.Module):
    '''
    Args:
        kernel_size: the kernel size of the 2d gaussian kernel
        sigma: [min_sigma, max_sigma], when creating new kernels, the module will
               randomly select a sigma from [min_sigma, max_sigma].
    '''
    def __init__(self, kernel_size: int, sigma=(0.1, 2), p=0.5, cuda=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma if not isinstance(sigma, numbers.Number) else (sigma, sigma)
        self.cuda = cuda
        self.p = p
        
    def _get_batch_gaussian_kernel2d(self, kernel_size: int, sigma=Union[float, list]):
        """generate multiple 2d gaussian kernel

        Args:
            kernel_size: the kernel size of the 2d gaussian kernel
            sigma: one or multiple sigma

        Returns:
            N 2d gaussian kernels, (N, kernel_size, kernel_size), where N is the
                length of sigma.
        """
        if isinstance(sigma, numbers.Number):
            sigma = [sigma]
        sigma = torch.Tensor(sigma)
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x[None, ...] / sigma[..., None]).pow(2))
        kernel1d = pdf / pdf.sum(dim=1)[:, None]
        kernel2d = torch.matmul(kernel1d[..., None], kernel1d[:, None, :])
        return kernel2d
    

    def gene_kernels(self, N):
        sigma = torch.empty(N).uniform_(self.sigma[0], self.sigma[1])
        kernels = self._get_batch_gaussian_kernel2d(self.kernel_size, sigma)
        # print("kernels size is ", kernels.shape)
        return kernels
    
    def gaussian_conv(self, x):
        B, C, H, W = x.shape
        if B < 1:
            return x
        assert B > 0, x.shape
        kernels = self.gene_kernels(B).unsqueeze(1)
        if self.cuda:
            kernels = kernels.cuda()
        x = x.transpose(0, 1)
        x = F.conv2d(x, kernels, padding="same", groups=B)
        x = x.transpose(0, 1)
        return x
    
    def conv_4D(self, x, label=None):
        B, C, H, W = x.shape
        p = torch.rand(B)
        if label is None:
            index1 = torch.where(p <= self.p)
        else:
            index1 = torch.where((p <= self.p) & (label == 1))
        index2 = torch.where(p > self.p)
        x_gaussian = self.gaussian_conv(x[index1])
        # x = torch.concat([x_gaussian, x[index2]], dim=0)
        x[index1] = x_gaussian
        return torch.clip(x.contiguous(), min=0., max=1.)
    
    def forward(self, x, label=None):
        if label is not None and label.is_cuda:
            label = label.cpu()
            
        if x.ndim == 4:
            return self.conv_4D(x, label)
        elif x.ndim == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, 'b c t h w -> b (c t) h w')
            x = self.conv_4D(x, label)
            x = rearrange(x, 'b (c t) h w -> b c t h w', c=C, t=T)
            return x
        return x

# + tags=["active-ipynb", "style-student"]
# blur = BatchGaussianBlur(kernel_size=5, sigma=(1, 10))
# x = torch.rand(32, 30, 224, 224)
# y = torch.rand(32, 3, 10, 224, 224)
# print(blur(x).shape, blur(y).shape)
#
# # 测试 带标签
# label = torch.empty(32, dtype=torch.long).random_(2)
# print(blur(x, label).shape, blur(y, label).shape)
