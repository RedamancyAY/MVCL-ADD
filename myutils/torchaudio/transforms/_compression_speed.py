# %%
# %load_ext autoreload
# %autoreload 2

# %%
import torchaudio
import torch
from torchaudio.io import AudioEffector
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
import warnings
import random

# %%
from hide_warnings import hide_warnings

# %%
try:
    from self_operation import RandomSpeed,RandomAudioClip
    from _audio_compression import RandomAudioCompression
except ImportError:    
    from .self_operation import RandomSpeed,RandomAudioClip
    from ._audio_compression import RandomAudioCompression


# %%
@dataclass
class RandomAudioCompressionSpeedChanging:
    def __init__(
        self,
        p_compression=0.5,
        sample_rate=16000,
        min_speed=0.5,
        max_speed=2.0,
        p_speed=1.0,
        audio_length=48000,
    ):
        """post initialization

        check the values of the min_speed and max_speed
        """

        self.compressor = RandomAudioCompression(
            p=p_compression, sample_rate=sample_rate
        )
        self.speed_changer = RandomSpeed(
            min_speed=min_speed, max_speed=max_speed, p=p_speed
        )

        self.audio_length = audio_length

    def __call__(self, x: torch.Tensor, metadata=None, **kwargs) -> torch.Tensor:

        target_speed = self.speed_changer.get_random_speed()
        
        need_length = int(self.audio_length * target_speed) + 10
        waveform_len = x.shape[1]
        if waveform_len > need_length:
            start = random.randint(0, waveform_len - need_length)
            x = x[:, start : start + need_length]

        x = self.compressor(x, metadata=metadata)
        x = self.speed_changer._random_speed(x, target_speed)
        if metadata is not None:
            metadata = self.speed_changer.set_speed_label(target_speed, metadata)
        return x

    def batch_apply(self, x: torch.Tensor):
        raise NotImplementedError
