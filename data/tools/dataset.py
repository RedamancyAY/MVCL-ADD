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

# + editable=true slideshow={"slide_type": ""}
"""Common preprocessing functions for audio data."""
import functools
import logging
import math
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import inspect
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.functional import apply_codec


# -

class WaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        sample_rate: int = 16_000,
        normalize: bool = True,
        transform=None,
        dtype='Tensor',
        read_features=[],
        **kwargs,
    ) -> None:
        super().__init__()

        self.data = data
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.transform = transform
        self.dtype = dtype.lower()
        self.read_features = read_features

    def read_metadata(self, index: int) -> dict:
        item = self.data.iloc[index]
        keys = item.keys()
        res = {"sample_rate": self.sample_rate}
        if "label" in keys:
            res["label"] = item["label"]
        if "name" in keys:
            res["name"] = item["name"]
        else:
            res["name"] = item["audio_path"]
        if "vocoder_label" in keys:
            res["vocoder_label"] = item["vocoder_label"]
        else:
            res["vocoder_label"] = 0

        if "vocoder_label_org" in keys:
            res["vocoder_label_org"] = item["vocoder_label_org"]
        
        
        res["speed_label"] = 5
        res["compression_label"] = 0

        if "emotion_label" in keys:
            res["emotion_label"] = item["emotion_label"]

        if "compression" in keys:
            res['compression'] = item['compression']
        if 'source' in keys:
            res['source'] = item['source']
        if 'language' in keys:
            res['language'] = item['language']

        for key in self.read_features:
            res[key] = item[key]
        
        return res

    def torch_load(self, path, fps):
        if fps != self.sample_rate:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
                path, [["rate", f"{self.sample_rate}"]], normalize=self.normalize
            )
        else:
            waveform, sample_rate = torchaudio.load(path, normalize=self.normalize)
        return waveform

    def numpy_load(self, path, fps):
        waveform, sample_rate = librosa.load(path, sr=self.sample_rate)
        waveform = waveform[None, :]
        return waveform


    def get_audio_length(self, index: int) -> int:
        item = self.data.iloc[index]
        if 'audio_len' in item.keys():
            return item['audio_len']
        elif 'audio_length' in item.keys():
            return item['audio_length']
        else:
            raise KeyError("when getting audio length, either the `audio_len` nor the `audio_length` in the item keys")
    
    def read_audio(self, index: int) -> Tuple[torch.Tensor, int, int]:
        item = self.data.iloc[index]

        path = item["audio_path"]
        fps = item["audio_fps"]

        if self.dtype == 'tensor':
            waveform = self.torch_load(path, fps)
        else:
            waveform = self.numpy_load(path, fps)

        return waveform

    def __getitem__(self, index: int):

        
        waveform = self.read_audio(index)
        res = self.read_metadata(index)

        if self.transform is not None:
            for t in self.transform:
                if 'metadata' in inspect.getfullargspec(t).args:
                    waveform = t(waveform, metadata=res)
                else:
                    waveform = t(waveform)

        res["audio"] = waveform
        # print(waveform.shape)
        return res

    def __len__(self) -> int:
        return len(self.data)

