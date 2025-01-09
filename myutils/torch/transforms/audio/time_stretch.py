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

import torch
from audiomentations import TimeStretch as Org_TimeStretch
import numpy as np


class TimeStretch:
    def __init__(self, min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=1.0, sample_rate=16000):
        self.transform = Org_TimeStretch(
            min_rate=min_rate,
            max_rate=max_rate,
            leave_length_unchanged=leave_length_unchanged,
            p=p,
        )
        self.sample_rate = sample_rate
        
    def __call__(self, audio, sample_rate=-1):
        if sample_rate == -1:
            sample_rate = self.sample_rate
        
        if type(audio) == np.ndarray:
            return self.transform(audio, sample_rate=sample_rate)
        else:
            audio = self.transform(audio.numpy(), sample_rate=sample_rate)
            return torch.from_numpy(audio)

# + tags=["active-ipynb"]
# my_waveform_ndarray = torch.rand(1, 16000)
# transform = TimeStretch()
# transform(my_waveform_ndarray).shape
