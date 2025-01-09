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

# +
import math
import random
from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torchaudio.functional import pitch_shift as pitchshift_transform
from torchaudio.functional import speed as speed_transform
from torchaudio.transforms import PitchShift


# -

class AudioToTensor:
    def __call__(self, x, *args, **kwargs):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise ValueError(
                f"Error, the input audio is not np.ndarray or torch.Tensor, but is {type(waveform)}"
            )


# ## Clip or pad audio

@dataclass
class RandomAudioClip:
    length: int

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError(f"Error, the audio length must > 0, but is {self.length}")

    def _pad_audio(self, waveform):
        waveform_len = waveform.shape[-1]
        num_repeats = int(math.ceil(self.length / waveform_len))
        if isinstance(waveform, np.ndarray):
            waveform = np.tile(waveform, (1, num_repeats))[:, : self.length]
        elif isinstance(waveform, torch.Tensor):
            waveform = torch.tile(waveform, (1, num_repeats))[:, : self.length]
        else:
            raise ValueError(
                f"Error, the input audio is not np.ndarray or torch.Tensor, but is {type(waveform)}"
            )
        return waveform

    def get_starting_point(self, waveform_len):
        start = random.randint(0, waveform_len - self.length)
        return start

    def __call__(self, waveform, *args, **kwargs):
        """
        waveform: (C, L)
        """

        waveform_len = waveform.shape[-1]

        if waveform_len >= self.length:
            start = self.get_starting_point(waveform_len)
            waveform = waveform[:, start : start + self.length]
        else:
            # the input waveform needs to be padded
            waveform = self._pad_audio(waveform)
        return waveform


class CentralAudioClip(RandomAudioClip):
    def get_starting_point(self, waveform_len):
        start = (waveform_len - self.length) // 2
        return start


# +
# x = np.random.randn(2, 5).astype(np.float32)
# t1 = RandomAudioClip(7)
# t2 = CentralAudioClip(3)
# t1(x), t1(torch.tensor(x)), t2(x)
# -

# ## Change Audio Speed

# +
# x = np.random.randn(1, 48000).astype(np.float32)
# y1, _ = speed_transform(torch.tensor(x), 16000, 1.1)
# print(y1.shape, y1)
# -

def _source_target_sample_rate(orig_freq: int, speed: float):
    source_sample_rate = int(speed * orig_freq)
    target_sample_rate = int(orig_freq)
    gcd = math.gcd(source_sample_rate, target_sample_rate)
    return source_sample_rate // gcd, target_sample_rate // gcd


# +
def torchaudio_resample(x, orig_freq, new_freq):
    x = torchaudio.functional.resample(
        x,
        orig_freq=orig_freq,
        new_freq=new_freq,
        resampling_method="sinc_interp_kaiser",
    )
    return x


def librosa_resample(x, orig_freq, new_freq):
    return librosa.resample(
        x, orig_sr=orig_freq, target_sr=new_freq, res_type="kaiser_fast"
    )


# -

class Torchaudio_resampler:
    def __init__(self):
        self.transforms = {}

    def function_resample(self, x, orig_freq, new_freq):
        x = torchaudio.functional.resample(
            x,
            orig_freq=orig_freq,
            new_freq=new_freq,
            resampling_method="sinc_interp_kaiser",
        )
        return x

    def transform_resample(self, x, orig_freq, new_freq):
        key = f"{orig_freq}-{new_freq}"
        if key not in self.transforms.keys():
            self.transforms[key] = T.Resample(
                orig_freq,
                new_freq,
                resampling_method="sinc_interp_kaiser",
            )

        x = self.transforms[key](x)
        return x


@dataclass
class RandomSpeed:
    min_speed: float = 0.5
    max_speed: float = 2.0
    p: float = 0.5

    def __post_init__(self):
        """post initialization

        check the values of the min_speed and max_speed

        """
        if self.min_speed <= 0:
            raise ValueError(
                f"Error, min speed must be > 0, your input is {self.min_speed}"
            )
        if self.min_speed > self.max_speed:
            raise ValueError(
                f"Error, min_speed must < max_speed, your input is {self.min_speed} and {self.max_speed}"
            )

        self.speed_to_label = {
            x / 10: x - int(self.min_speed * 10)
            for x in range(int(self.min_speed * 10), int(self.max_speed * 10) + 1, 1)
        }

        self.resampler = Torchaudio_resampler()

    def _random_speed(self, x, speed):
        if speed == 1.0:
            return x
        else:
            orig_freq, new_freq = _source_target_sample_rate(16000, speed)

            # x, _ = speed_transform(x, 16000, speed)
            if isinstance(x, np.ndarray):
                x = librosa_resample(x, orig_freq, new_freq)
            elif isinstance(x, torch.Tensor):
                x = self.resampler.function_resample(x, orig_freq, new_freq)
                # x = self.resampler.transform_resample(x, orig_freq, new_freq)
            else:
                raise ValueError(
                    f"Error, the input audio is not np.ndarray or torch.Tensor, but is {type(x)}"
                )

            return x

    def get_random_speed(self):
        if np.random.rand() > self.p:
            target_speed = 1.0
        else:
            target_speed = np.random.rand() * (self.max_speed - self.min_speed) + self.min_speed
            target_speed = round(target_speed, 1)
        return target_speed

    def set_speed_label(self, target_speed, metadata):
        metadata["speed_label"] = self.speed_to_label[target_speed]
        metadata["speed"] = target_speed
        return metadata

    def __call__(self, x: torch.Tensor, metadata=None, **kwargs) -> torch.Tensor:
        if x.ndim not in [1, 2]:
            raise ValueError("Error, input audio should be (L), or (C, L)", x.shape)

        if np.random.rand() > self.p:
            target_speed = 1.0
        else:
            target_speed = (
                np.random.rand() * (self.max_speed - self.min_speed) + self.min_speed
            )
            target_speed = round(target_speed, 1)
        if metadata is not None:
            metadata["speed_label"] = self.speed_to_label[target_speed]
            metadata["speed"] = target_speed
        x = self._random_speed(x, target_speed)
        # print(target_speed, x.shape, self.speed_to_label[target_speed])
        return x

    def batch_apply(self, x: torch.Tensor):
        batch_size = x.shape[0]
        labels = [self.get_random_speed() for i in range(batch_size)]
        for i, speed in enumerate(labels):
            x[i] = self._random_speed(x[i], speed)
        return x, labels


# +
# t = RandomSpeed(p=1.0)
# x = torch.randn(1, 48000)
# y = x.numpy()
# for i in range(100):
#     t(x), t(y)
# -

# ## PitchShift

@dataclass
class RandomPitchShift:
    min_pitch: float = -6
    max_pitch: float = 6
    p: float = 0.5

    def __post_init__(self):
        """post initialization

        check the values of the min_pitch and max_pitch

        """
        if self.min_pitch > self.max_pitch:
            raise ValueError(
                f"Error, min_pitch must < max_pitch, your input is {self.min_pitch} and {self.max_pitch}"
            )

        self.transforms = {
            x: PitchShift(sample_rate=16000, n_steps=x)
            for x in range(self.min_pitch, self.max_pitch + 1, 1)
        }

        self.pitch_to_label = {
            x: x - self.min_pitch for x in range(self.min_pitch, self.max_pitch + 1, 1)
        }

    def _random_pitch(self, x, pitch, sr=16000):
        if pitch == 0:
            return x

        # x = pitchshift_transform(x, 16000, pitch, n_fft=256)
        # x = self.transforms[pitch](x)
        # return x

        # print(x.device, pitch)

        if isinstance(x, np.ndarray):
            x = librosa.effects.pitch_shift(x, sr=sr, n_steps=pitch)
        elif isinstance(x, torch.Tensor):
            # x = pitchshift_transform(x, 16000, pitch, n_fft=512)
            x = librosa.effects.pitch_shift(x.numpy(), sr=sr, n_steps=pitch)
            x = torch.from_numpy(x)
        else:
            raise ValueError(
                f"Error, the input audio is not np.ndarray or torch.Tensor, but is {type(x)}"
            )

        return x

    def __call__(self, x: torch.Tensor, metadata=None, **kwargs) -> torch.Tensor:
        if x.ndim not in [1, 2]:
            raise ValueError("Error, input audio should be (L), or (C, L)", x.shape)

        if np.random.rand() > self.p:
            target_pitch = 0
        else:
            target_pitch = (
                np.random.rand() * (self.max_pitch - self.min_pitch) + self.min_pitch
            )
            target_pitch = int(target_pitch)

        sr = 16000
        if metadata is not None:
            metadata["pitch"] = self.pitch_to_label[target_pitch]
            if "speed" in metadata.keys():
                sr = sr / metadata["speed"]

        x = self._random_pitch(x, target_pitch, sr=sr)
        # print(target_pitch, x.shape, self.pitch_to_label[target_pitch])
        return x

# +
# t = RandomPitchShift(p=1.0, min_pitch=-6, max_pitch=6)
# x = torch.randn(1, 48000)
# y = x.numpy()
# for i in range(10):
#     t(y)

# +
# import librosa
# x = torch.randn(1, 48000)
# y = x.numpy()
# y1 = librosa.effects.pitch_shift(y, sr=16000, n_steps=4)

# from main import pitch_shift
# y2 = pitch_shift(x[None,...], 4, sample_rate=16000)
# -


