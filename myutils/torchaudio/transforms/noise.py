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

# %load_ext autoreload
# %autoreload 2

# +
import math
import os
import pathlib
import random

import numpy as np
import torch
import torchaudio


# -

class RandomNoise:
    def __init__(self, snr_min_db: float = -10.0, snr_max_db: float = 100.0, p: float = 0.5):
        super(RandomNoise, self).__init__()

        self.p = p
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db

    def torch_noise(self, signal, target_snr):
        signal_watts = torch.mean(signal**2)
        signal_db = 10 * torch.log10(signal_watts)

        noise_db = signal_db - target_snr
        noise_watts = 10 ** (noise_db / 10)
        noise = torch.normal(0.0, noise_watts.item() ** 0.5, signal.shape)
        return noise

    def numpy_noise(self, signal, target_snr):
        signal_watts = np.mean(signal**2)
        signal_db = 10 * np.log10(signal_watts)

        noise_db = signal_db - target_snr
        noise_watts = 10 ** (noise_db / 10)
        noise = np.random.normal(0.0, noise_watts.item() ** 0.5, signal.shape)
        return noise.astype(np.float32)

    def random_noise(self, signal: torch.Tensor, snr) -> torch.Tensor:
        signal_watts = torch.mean(signal**2)
        signal_db = 10 * torch.log10(signal_watts)

        noise_db = signal_db - target_snr
        noise_watts = 10 ** (noise_db / 10)
        noise = torch.normal(0.0, noise_watts.item() ** 0.5, signal.shape)

        output = signal + noise

        return output

    def __call__(self, x: torch.Tensor, metadata=None, **kwargs) -> torch.Tensor:
        assert x.ndim in [1, 2], "input audio should be (L), or (C, L)"

        if np.random.rand() > self.p:
            return x

        target_snr = np.random.rand() * (self.snr_max_db - self.snr_min_db + 1.0) + self.snr_min_db

        if isinstance(x, np.ndarray):
            noise = self.numpy_noise(x, target_snr)
        elif isinstance(x, torch.Tensor):
            noise = self.torch_noise(x, target_snr)
        else:
            raise ValueError(f"Error, the input audio is not np.ndarray or torch.Tensor, but is {type(x)}")

        return x + noise


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# %timeit
# noise = RandomNoise(p=1)
# x = torch.randn(1, 48000)
# y1, y2 = noise(x), noise(x.numpy())
# y1 - y2
# -

# ## Gaussian Noise

class AddGaussianSNR:
    def __init__(self, snr_min_db: float = -10.0, snr_max_db: float = 100.0, p: float = 0.5):
        super().__init__()

        self.p = p
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db

    def random_noise(self, signal: torch.Tensor) -> torch.Tensor:
        assert signal.ndim == 3
        B = signal.shape[0]
        target_snr = torch.rand(B) * (self.snr_max_db - self.snr_min_db + 1.0) + self.snr_min_db

        signal_watts = torch.mean(signal**2, dim=[1, 2])
        signal_db = 10 * torch.log10(signal_watts)

        noise_db = signal_db - target_snr.to(signal.device)
        noise_watts = 10 ** (noise_db / 10)
        noise_watts = noise_watts**0.5
        noise = torch.normal(0.0, torch.ones_like(signal) * noise_watts[..., None, None])

        output = signal + noise

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, "input audio should be (B, C, L)"

        B = x.shape[0]
        p = np.random.rand(B)
        tar_index = np.nonzero(p < self.p)[0]

        if len(tar_index) == 0:
            return x

        x[tar_index] = self.random_noise(x[tar_index])
        return x


# +
x = torch.randn(10, 1, 48000)

module = AddGaussianSNR(p=0.5, snr_min_db=1, snr_max_db=20)
module(x)
# -

# ## Random Background Noise



class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, noise_type='bg', min_snr_db=0, max_snr_db=15, p=0.5):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.noise_type = noise_type
        self.p = p

        if not os.path.exists(noise_dir):
            raise IOError(f"Noise directory `{noise_dir}` does not exist")
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob("**/*.wav"))
        if len(self.noise_files_list) == 0:
            raise IOError(f"No .wav file found in the noise directory `{noise_dir}`")

    def __call__(self, audio_data):
        if random.random() > self.p:
            return audio_data

        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ["remix", "1"],  # convert to mono
            ["rate", str(self.sample_rate)],  # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)

        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length - audio_length)
            noise = noise[..., offset : offset + audio_length]
        elif noise_length < audio_length:
            noise = torch.cat(
                [noise, torch.zeros((noise.shape[0], audio_length - noise_length))],
                dim=-1,
            )

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)

        if self.noise_type != 'bg':    
            noise = get_gaussian_noise(audio_data, snr_db)
            return audio_data + noise
        else:
            return _add_noise(audio_data, 48000, noise, snr_db)
        
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power
        
        return (scale * audio_data + noise) / 2
        return random_noise2(audio_data, noise, snr_db)


def _add_noise(speech_sig, vad_duration, noise_sig, snr):
    """add noise to the audio.
    :param speech_sig: The input audio signal (Tensor).
    :param vad_duration: The length of the human voice (int).
    :param noise_sig: The input noise signal (Tensor).
    :param snr: the SNR you want to add (int).
    :returns: noisy speech sig with specific snr.
    """
    if vad_duration != 0:
        snr = 10**(snr/10.0)
        speech_power = torch.sum(speech_sig**2)/vad_duration
        noise_power = torch.sum(noise_sig**2)/noise_sig.shape[1]
        noise_update = noise_sig / torch.sqrt(snr * noise_power/speech_power)

        # print(torch.sqrt(snr * noise_power/speech_power))
        
        if speech_sig.shape[1] > noise_update.shape[1]:
            # padding
            temp_wav = torch.zeros(1, speech_sig.shape[1])
            temp_wav[0, 0:noise_update.shape[1]] = noise_update
            noise_update = temp_wav
        else:
            # cutting
            noise_update = noise_update[0, 0:speech_sig.shape[1]]

        return noise_update + speech_sig
    
    else:
        return speech_sig


def random_noise2(signal, noise, target_snr) -> torch.Tensor:
    signal_watts = torch.mean(signal**2)
    signal_db = 10 * torch.log10(signal_watts)

    noise_db = signal_db - target_snr
    noise_watts = 10 ** (noise_db / 10)
    noise = torch.normal(0.0, noise_watts.item() ** 0.5, signal.shape)

    # print(signal_db, noise_db, torch.mean(noise-signal), target_snr)
    
    output = signal + noise

    return output


def get_gaussian_noise(signal, target_snr) -> torch.Tensor:
    signal_watts = torch.mean(signal**2)
    signal_db = 10 * torch.log10(signal_watts)

    noise_db = signal_db - target_snr
    noise_watts = 10 ** (noise_db / 10)
    noise = torch.normal(0.0, noise_watts.item() ** 0.5, signal.shape)
    return noise


# +
def calculate_rms(audio):
    return torch.sqrt(torch.mean(audio**2))

def add_noise_to_audio(audio, noise, snr):
    # Calculate RMS of the audio and the noise
    rms_audio = calculate_rms(audio)
    rms_noise = calculate_rms(noise)

    
    # Calculate desired RMS of the noise
    desired_rms_noise = rms_audio / (10**(snr / 20))

    # Scale noise to achieve desired RMS
    scaled_noise = noise * (desired_rms_noise / rms_noise)

    # print(rms_audio, rms_noise, snr, desired_rms_noise, (desired_rms_noise / rms_noise))
    
    # Add scaled noise to the audio
    noisy_audio = audio + scaled_noise

    noisy_audio = noisy_audio / torch.max(torch.abs(noisy_audio))
    
    return noisy_audio

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# # bg_noise = RandomBackgroundNoise(
# #     16000, noise_dir="/home/ay/data/0-原始数据集/musan/noise", p=1.0, min_snr_db=5, max_snr_db=5
# # )
# # x = torch.randn(1, 48000)
# # bg_noise(x) - x
