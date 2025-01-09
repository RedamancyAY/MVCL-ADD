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

# `R(2+1)D`是论文A Closer Look at Spatiotemporal Convolutions for Action Recognition提出的卷积方式：将3D卷积分解为独立的空间和时间卷积。
#
# 代码参考：[Github](https://github.com/fourierer/Video_Classification_ResNet3D_R2plus1D_ip-CSN_train-UCF101-HMDB51-Kinetics400-from-scratch/blob/master/train/models/resnet2p1d.py)

# <center><img src="https://cdn.jsdelivr.net/gh/RedamancyAY/CloudImage@main/img/CleanShot 2022-10-24 at 09.46.11@2x.png" width="600" alt="Residual network architectures for video classification considered in this work. (a) R2D are 2D ResNets； (b) MCx are ResNets with mixed convolutions (MC3 is presented in this figure)； (c) IMCx use reversed mixed convolutions (rMC3 is shown here); (d) R3D are 3D ResNets; and (e) R(2+1)D are ResNets with (2+1)D convolutions. For interpretability, residual connections are omitted."/></center>

# 实现方法为：将3D卷积的$N_i$大小为$N_{i-1}\times t \times d \times d$个滤波分解为
# 1. $M_i$个2D卷积，滤波大小为$N_{i-1}\times 1 \times d \times d$
# 2. $N_i$个时间卷积，滤波大小为$M_i\times t \times 1\times 1$
#
# 其中，$M_i$和$N_i$的关系是：
# $$
# M_i=\left\lfloor\frac{t d^2 N_{i-1} N_i}{d^2 N_{i-1}+t N_i}\right\rfloor
# $$
# 这样可以使`R(2+1)D`和3D卷积的参数大致一样。
#

# + tags=[]
import torch
import torch.nn as nn


def conv_spatial(in_channels, mid_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv3d(
        in_channels,
        mid_channels,
        kernel_size=(1, kernel_size, kernel_size),
        stride=(1, stride, stride),
        padding=(0, padding, padding),
        bias=False,
    )


def conv_time(mid_channels, channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv3d(
        mid_channels,
        channels,
        kernel_size=(kernel_size, 1, 1),
        stride=(stride, 1, 1),
        padding=(padding, 0, 0),
        bias=False,
    )


class Conv2p1D(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        mid_channels=-1,
        t_kernel=3,
        t_stride=1,
        t_padding=1,
        s_kernel=3,
        s_stride=1,
        s_padding=1,
        bn=False,
    ):
        super().__init__()

        if mid_channels == -1:
            n_conv3D_para = in_channels * channels * (t_kernel * s_kernel * s_kernel)
            n_conv2p1D_para = in_channels * (s_kernel**2) + t_kernel * channels
            mid_channels = n_conv3D_para // n_conv2p1D_para

        self.conv_spatial = conv_spatial(
            in_channels,
            mid_channels,
            kernel_size=s_kernel,
            stride=s_stride,
            padding=s_padding,
        )
        self.conv_time = conv_time(
            mid_channels,
            channels,
            kernel_size=t_kernel,
            stride=t_stride,
            padding=t_padding,
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn = None if bn == False else nn.BatchNorm3d(mid_channels)

    def forward(self, x):
        x = self.conv_spatial(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv_time(x)
        return x
# -

# ## 测试 Conv(2+1)D 和 Conv3D的参数差距 

# + tags=["active-ipynb"]
# model = Conv2p1D(in_channels=3, channels=32, t_kernel=3, s_kernel=7, bn=False)
#
# for x in model.parameters():
#     print(x.shape)
#
# model2 = nn.Conv3d(3, 32, kernel_size=(3, 7, 7), bias=False)
# for x in model2.parameters():
#     print(x.shape)
#
# sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model2.parameters())

# + tags=["active-ipynb"]
# model = Conv2p1D(
#     in_channels=3,
#     channels=32,
#     t_kernel=3,
#     t_stride=1,
#     t_padding=1,
#     s_kernel=7,
#     s_stride=2,
#     s_padding=3,
# )
# x = torch.Tensor(5, 3, 15, 128, 128)
# model(x).shape
