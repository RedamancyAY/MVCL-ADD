# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import torch
import torchaudio

from .functional.lfcc import extract_lfcc, linear_fbank


# + tags=["active-ipynb", "style-activity"] editable=true slideshow={"slide_type": ""}
# from functional.lfcc import extract_lfcc, linear_fbank
# -

# 来自：https://github.com/ADDchallenge/FAD/blob/80ddee6f5d53f0bf163c6c98183d81dfbf876141/FAD_upload/lfcc-lcnn/g_lfcc_final.py

class LFCC:
    def __init__(
        self,
    ):
        self.lfcc_fb = linear_fbank()

    def __call__(self, wave):
        dtype = type(wave)
        spec = extract_lfcc(wave, self.lfcc_fb)
        spec = spec.astype(np.float32)

        if dtype is torch.Tensor:
            return torch.tensor(spec)
        return spec

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# x = torch.randn(13,1, 48000)
# module = LFCC()
# print(module(x).shape)
#
# from torchaudio.transforms import LFCC as LFCC2
#
# module = LFCC2(
#     n_lfcc=60,
#     speckwargs={"n_fft": 400, "hop_length": 160, "center": False},
# )
#
# module(x).shape
