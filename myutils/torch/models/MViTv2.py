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
# copied from https://github.com/ControlNet/LAV-DF/blob/9af5570da520233166a21eba4a495b66a1eb9186/model/video_encoder.py#L158

# +
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import LeakyReLU, Linear, MaxPool3d, Module, Sequential
from torchvision.models.video.mvit import MSBlockConfig, _mvit


# +
def generate_config(blocks, heads, channels, out_dim):
    num_heads = []
    input_channels = []
    kernel_qkv = []
    stride_q = [[1, 1, 1]] * sum(blocks)
    blocks_cum = np.cumsum(blocks)
    stride_kv = []

    for i in range(len(blocks)):
        num_heads.extend([heads[i]] * blocks[i])
        input_channels.extend([channels[i]] * blocks[i])
        kernel_qkv.extend([[3, 3, 3]] * blocks[i])

        if i != len(blocks) - 1:
            stride_q[blocks_cum[i]] = [1, 2, 2]

        stride_kv_value = 2 ** (len(blocks) - 1 - i)
        stride_kv.extend([[1, stride_kv_value, stride_kv_value]] * blocks[i])

    return {
        "num_heads": num_heads,
        "input_channels": [input_channels[0]] + input_channels[:-1],
        "output_channels": input_channels[:-1] + [out_dim],
        "kernel_q": kernel_qkv,
        "kernel_kv": kernel_qkv,
        "stride_q": stride_q,
        "stride_kv": stride_kv,
    }


def build_mvit(config, kwargs, temporal_size=512, spatial_size=(96, 96)):
    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )
    return _mvit(
        spatial_size=spatial_size, # must be (96, 96), otherwise output T > temporal_size
        temporal_size=temporal_size,
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        weights=None,
        progress=False,
        patch_embed_kernel=(3, 15, 15),
        patch_embed_stride=(1, 12, 12),
        patch_embed_padding=(1, 3, 3),
        **kwargs,
    )


def mvit_v2_b(out_dim: int, temporal_size: int, spatial_size, **kwargs):
    config = generate_config([2, 3, 16, 3], [1, 2, 4, 8], [96, 192, 384, 768], out_dim)
    return build_mvit(
        config, kwargs, temporal_size=temporal_size, spatial_size=spatial_size
    )


def mvit_v2_s(out_dim: int, temporal_size: int, spatial_size, **kwargs):
    config = generate_config([1, 2, 11, 2], [1, 2, 4, 8], [96, 192, 384, 768], out_dim)
    return build_mvit(
        config, kwargs, temporal_size=temporal_size, spatial_size=spatial_size
    )


def mvit_v2_t(out_dim: int, temporal_size: int, spatial_size, **kwargs):
    config = generate_config([1, 2, 5, 2], [1, 2, 4, 8], [96, 192, 384, 768], out_dim)
    return build_mvit(
        config, kwargs, temporal_size=temporal_size, spatial_size=spatial_size
    )


# -

class MvitVideoEncoder(nn.Module):
    def __init__(
        self,
        n_features: int = 256,
        temporal_size: int = 512,
        mvit_type: Literal["mvit_v2_t", "mvit_v2_s", "mvit_v2_b"] = "mvit_v2_t",
        spatial_size=(96, 96),
    ):
        super().__init__()
        if mvit_type == "mvit_v2_t":
            self.mvit = mvit_v2_t(n_features, temporal_size, spatial_size)
        elif mvit_type == "mvit_v2_s":
            self.mvit = mvit_v2_s(n_features, temporal_size, spatial_size)
        elif mvit_type == "mvit_v2_b":
            self.mvit = mvit_v2_b(n_features, temporal_size, spatial_size)
        else:
            raise ValueError(f"Invalid mvit_type: {mvit_type}")
        del self.mvit.head

    def forward(self, video: Tensor) -> Tensor:
        feat = self.mvit.conv_proj(video)
        thw = (feat.shape[2], feat.shape[3], feat.shape[4])
        # print("feat shape: ", feat.shape)
        feat = feat.flatten(2).transpose(1, 2)
        feat = self.mvit.pos_encoding(feat)
        # thw = (
        #     self.mvit.pos_encoding.temporal_size,
        # ) + self.mvit.pos_encoding.spatial_size
        # thw = (5,18,18)
        # print(thw)
        for block in self.mvit.blocks:
            feat, thw = block(feat, thw)
            # print(feat.shape, thw)

        feat = self.mvit.norm(feat)
        feat = feat[:, 1:]
        feat = feat.permute(0, 2, 1)
        return feat

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# s = 96
# t = 52
# model = MvitVideoEncoder(n_features=512, temporal_size=t, spatial_size=(s,s))
#
# x = torch.rand(4, 3, t // 2, s,s)
# model(x).shape
# -


