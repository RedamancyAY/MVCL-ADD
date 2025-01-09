import torchaudio
import torch
from torchaudio.io import AudioEffector
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
import warnings

from hide_warnings import hide_warnings


@dataclass
class RandomAudioCompression:
    def __init__(self, p=0.5, sample_rate=16000):
        """post initialization

        check the values of the min_speed and max_speed

        """

        self.sample_rate = sample_rate
        self.bit_rate = [16000, 32000, 64000]
        self.format_codec = {
            "mp4": "aac",
            "ogg": "opus",
            "mp3": "mp3",
        }
        self.p = p

        self.effectors = {}
        for _format, _codec in self.format_codec.items():
            for bit_rate in self.bit_rate:
                key = self.get_setting_name(_codec, bit_rate)
                _codec = None if _codec == 'mp3' else _codec
                self.effectors[key] = AudioEffector(
                    format=_format,
                    encoder=_codec,
                    codec_config=torchaudio.io.CodecConfig(bit_rate=bit_rate),
                )

        self.setting_to_label = {
            key: i + 1 for i, key in enumerate(self.effectors.keys())
        }
        self.setting_to_label["None"] = 0

    def get_setting_name(self, codec, bit_rate):
        key = f"{codec}_{bit_rate}"
        return key

    @hide_warnings
    def _compression(self, x, compression_setting):
        if compression_setting == "None":
            return x
        x = x.transpose(0, 1) # (C, L) => (L, C)
        L = x.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self.effectors[compression_setting].apply(x, sample_rate=self.sample_rate)[:L,]
        x = x.transpose(0, 1) # (L, C) => (C, L)
        return x

    def get_random_compression_setting(self):
        if np.random.rand() > self.p:
            compression_setting = "None"
        else:
            settings = list(self.effectors.keys())
            id = int(np.random.randint(0, len(settings), 1))
            compression_setting = settings[id]
        return compression_setting

    def __call__(self, x: torch.Tensor, metadata=None, **kwargs) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("Error, input audio should be (C, L), but is ", x.shape)

        compression_setting = self.get_random_compression_setting()

        if metadata is not None:
            metadata["compression_label"] = self.setting_to_label[compression_setting]

        x = self._compression(x, compression_setting)
        return x

    def batch_apply(self, x: torch.Tensor):
        batch_size = x.shape[0]
        settings = [self.get_random_compression_setting() for _ in range(batch_size)]
        labels = [self.setting_to_label[s] for s in settings]
        for i, _setting in enumerate(settings):
            x[i] = self._random_speed(x[i], _setting)
        return x, labels
