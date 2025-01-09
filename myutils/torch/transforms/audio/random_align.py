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

# +
import math
import os
import pathlib
import random

import torch
import torchaudio
# -

random.randint(4500, 18000)


class RandomAlign:
    def __init__(self, p=0.5, max_length=18000, min_length=4500):
        self.p = p
        self.max_length = max_length
        self.min_length = min_length

    def __call__(self, audio_data):
        if random.random() > self.p:
            return audio_data
        
        L = audio_data.shape[-1]
        align_length = random.randint(self.min_length, self.max_length)
        align = torch.zeros(1, align_length)
        if random.random() > 0.5:
            audio_data = torch.concat([audio_data[:, align_length:], align], dim=-1)
        else:
            audio_data = torch.concat([align, audio_data[:, 0:L-align_length]], dim=-1)
        return audio_data

# + tags=["active-ipynb"]
# Align = RandomAlign(p=1, min_length=2, max_length=4)
# audio_data = torch.arange(0, 6)[None, ...]
# transformed_audio = Align(audio_data)
# print(audio_data, transformed_audio)
