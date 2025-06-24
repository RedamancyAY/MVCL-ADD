"""
resnet for 1-d signal data, pytorch version
 
Shenda Hong, Oct 2019
"""

# +
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm

# -

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        downsample_stride,
        groups,
        downsample,
        use_bn,
        use_dropout,
        is_first_block=False,
    ):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        # self.stride = downsample_stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = downsample_stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups,
        )

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_dropout:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_dropout:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class ResNet1D(nn.Module):
    """


    model = ResNet1D(
        in_channels=1,
        base_filters=64,
        kernel_size=3,
        downsample_stride=2,
        groups=1,
        n_block=16,
        n_classes=2,
        downsample_gap=4,
        verbose=1,
    )


    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        downsample_stride: stride of kernel moving in downsample layer
        groups: set larger to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    """

    def __init__(
        self,
        in_channels,
        base_filters,
        kernel_size,
        downsample_stride,
        groups,
        n_block,
        n_classes,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_dropout=True,
        verbose=False,
    ):
        super(ResNet1D, self).__init__()

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.downsample_stride = downsample_stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=10,
            stride=4,
        )

        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.first_max_pool = nn.MaxPool1d(5, 4, padding=1)
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block > 0 and i_block % self.downsample_gap == 0:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                downsample_stride=self.downsample_stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_dropout=self.use_dropout,
                is_first_block=is_first_block,
            )
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.rnn = nn.LSTM(512, 512 // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        if n_classes > 0:
            self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = x

        # first conv
        if self.verbose:
            print("input shape", out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print("after first conv", out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_max_pool(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print(
                    "i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}".format(
                        i_block, net.in_channels, net.out_channels, net.downsample
                    )
                )
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        # if self.verbose:
        #     print('final pooling', out.shape)
        # # out = self.do(out)
        # out = self.dense(out)
        # if self.verbose:
        #     print('dense', out.shape)
        # # out = self.softmax(out)
        # if self.verbose:
        #     print('softmax', out.shape)

        return out

    def _compute_blocks(self, s, e, out):
        for i_block in range(s, e):
            net = self.basicblock_list[i_block]
            out = net(out)
        return out

    def compute_stage1(self, out):
        if self.verbose:
            print("ResNet1D input shape", out.shape)

        out = self.first_block_conv(out)
        out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_max_pool(out)

        out = self._compute_blocks(0, self.n_block // 4, out)
        if self.verbose:
            print("ResNet1D Stage 1: output shape", out.shape)
        return out

    def compute_stage2(self, out):
        out = self._compute_blocks(self.n_block // 4, self.n_block // 2, out)
        if self.verbose:
            print("ResNet1D Stage 2: output shape", out.shape)
        return out

    def compute_stage3(self, out):
        out = self._compute_blocks(self.n_block // 2, self.n_block // 4 * 3, out)
        if self.verbose:
            print("ResNet1D Stage 3: output shape", out.shape)
        return out

    def compute_stage4(self, out):
        out = self._compute_blocks(self.n_block // 4 * 3, self.n_block, out)
        if self.verbose:
            print("ResNet1D Stage 4: output shape", out.shape)
        return out

    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10
        code = torch.div(code, code_norm)
        return code
        
        # feat = code
        # feat = feat / (1e-9 + torch.norm(feat, p=2, dim=-1, keepdim=True))
        
        # return feat

    def compute_latent_feature(self, out):
        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out) # [64, 512, 47]

        # out = self.rnn(out.transpose(1, 2))[0].transpose(1, 2)
        
        
        out = out.mean(-1)

        out = self.feature_norm(out)

        if self.verbose:
            print("ResNet1D Latent Feature: output shape", out.shape)

        return out


# +
# model = ResNet1D(
#     in_channels=1,
#     base_filters=64,
#     kernel_size=3,
#     downsample_stride=4,
#     groups=1,
#     n_block=8,
#     n_classes=0,
#     downsample_gap=2,
#     increasefilter_gap=2,
#     verbose=1,
# )
# x = torch.randn(2, 1, 48000)
# model(x)
