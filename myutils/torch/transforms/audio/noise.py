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
import math
import os
import pathlib
import random

import torch
import torchaudio


# -

class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=15, p=0.5):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

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
        noise, _ = torchaudio.sox_effects.apply_effects_file(
            random_noise_file, effects, normalize=True
        )
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
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise) / 2

# + tags=["active-ipynb"]
# noise_transform = RandomBackgroundNoise(
#     sample_rate=16000, noise_dir="/home/ay/musan/noise"
# )
# audio_data = torch.rand(1, 10000)
# transformed_audio = noise_transform(audio_data)
