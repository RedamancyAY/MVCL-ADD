# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + editable=true slideshow={"slide_type": ""}
# %load_ext autoreload
# %autoreload 2

# + editable=true slideshow={"slide_type": ""}
import functools
import logging
import math
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.functional import apply_codec
from torchvision.io import read_video

# + editable=true slideshow={"slide_type": ""}
from myutils.tools.audio.read_info import read_audio_fps
from myutils.tools.video.read import (
    read_video_clip_cv2,
    read_video_clip_from_frames,
    read_video_from_frames,
)


# -

# # Audio Operations

def trim_audio_salience(waveform, sample_rate):
    """
    Trim the salience in a audio
    Args:
        waveform: (C, L) or (L), the waveform of the input audio
        sample_rate: the sample rate of the input audio

    Returns:
        waveform, sample_rate
    """

    SOX_SILENCE = [
        # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
        # from beginning and middle/end
        ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
    ]

    (
        waveform_trimmed,
        sample_rate_trimmed,
    ) = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, SOX_SILENCE)

    if waveform_trimmed.size()[1] > 0:
        waveform = waveform_trimmed
        sample_rate = sample_rate_trimmed
    return waveform, sample_rate


def read_audio_with_fps(
    audio_path: str,
    audio_fps: int = None,
    target_fps: int = 16_000,
    normalize: bool = True,
):
    """
    Read audio with specified sample rate (fps).
    Args:
        audio_path: the path of the audio file
        audio_fps: the fps of the input audio. if None, read the fps from audio_path
        target_fps: the wanted fps
        normalize: whether normalize the audio in reading

    Returns:
        waveform, sample_rate
    """

    audio_fps = audio_fps or read_audio_fps(audio_path)
    if audio_fps != target_fps:
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
            audio_path, [["rate", f"{target_fps}"]], normalize=normalize
        )
    else:
        waveform, sample_rate = torchaudio.load(audio_path, normalize=normalize)
    return waveform, sample_rate


# # Dataset

def deal_possible_error(func):
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
        except Exception as e:
            return None
        return res
    return wrapper


# ## Sample-based transform Dataset

class SampleTransformDataset(torch.utils.data.Dataset):
    """
    Apple transform for each sample of the dataset. It will apple transform at the end of
    the `__get_item__()` method.
    """

    def __init__(self, transforms=None):
        """
        Args:
            transforms: a dict, {'key1':transform1, 'key2':transform2}
        """
        self.transforms = transforms

    def apply_transforms(self, res):
        # print(self.transforms)
        if self.transforms is None:
            return res

        try:
            for key in self.transforms:
                # print(key)
                res[key] = self.transforms[key](res[key])
        except KeyError:
            raise KeyError("Your transforms do not have the key", key)
        return res


# ### Audio-Visual Dataset

# + editable=true slideshow={"slide_type": ""}
class AudioVisualDataset(SampleTransformDataset):
    """Torch dataset to load data from a provided directory.

    Args:
        data: A pandas dataframe, whose 'path' column is the path for each audio file.
        sample_rate: The used sample rate for the audio
        amount: default None. If not none, it means the number of used audio
        normalize: default True.
        trim: default True. trim all silence that is longer than 0.2s and louder than 1% volume
        phone_call: default False.
    Raises:
        IOError: If the directory does ot exists or the directory did not contain any wav files.
    """

    def __init__(
        self,
        data,
        transforms=None,
        # audio setting
        audio_fps: int = 16_000,
        audio_normalize: bool = True,
        audio_trim: bool = False,
        # video setting
        has_extracted_video_frames=False,
        has_resized_video=False,
        video_size=96,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(transforms=transforms)

        self.data = data
        self.audio_fps = audio_fps
        self.audio_normalize = audio_normalize
        self.audio_trim = audio_trim
        self.has_extracted_video_frames = has_extracted_video_frames
        self.has_resized_video = has_resized_video
        self.video_size = video_size

    def read_metadata(self, index: int) -> dict:
        item = self.data.iloc[index]
        keys = item.keys()
        res = {}
        res.update({"audio_fps": self.audio_fps})
        res.update({"video_fps": item["video_fps"]})
        res.update({"video_path": item["video_path"]})
        res.update({"audio_path": item["audio_path"]})
        if self.has_extracted_video_frames:
            res["video_img_path"] = item["video_img_path"]
        return res

    def read_audio(self, index: int) -> Tuple[torch.Tensor, int, int]:
        item = self.data.iloc[index]

        audio_path = item["audio_path"]
        audio_fps = item["audio_fps"]

        waveform, sample_rate = read_audio_with_fps(
            audio_path, audio_fps=audio_fps, target_fps=self.audio_fps
        )

        if self.audio_trim:
            waveform = trim_audio_salience(waveform, sample_rate)
        return waveform

    def read_video(self, index: int) -> Tuple[torch.Tensor, int, int]:
        item = self.data.iloc[index]

        if self.has_extracted_video_frames:
            video = read_video_from_frames(item["video_img_path"])
        else:
            if self.has_resized_video:
                h = self.video_size
                video_path = item["video_path"].replace('.mp4', f'-{h}x{h}.mp4')
            else:
                video_path = item["video_path"]
            video = read_video_clip_cv2(video_path)

        return video

    @deal_possible_error
    def __getitem__(self, index: int) -> dict:
        res = self.read_metadata(index)
        # res['audio'] = torch.ones(2,2)
        # res['video'] = torch.ones(2,2)
        res["audio"] = self.read_audio(index)
        res["video"] = self.read_video(index)
        res = self.apply_transforms(res)
        return res

    def __len__(self) -> int:
        return len(self.data)


# -

# ### Audio-Visual Random Clip Dataset

# + editable=true slideshow={"slide_type": ""}
class AudioVisualRandomClipDataset(AudioVisualDataset):
    """Torch dataset to load data from a provided directory.

    Args:
        data: A pandas dataframe, whose 'path' column is the path for each audio file.
        sample_rate: The used sample rate for the audio
        amount: default None. If not none, it means the number of used audio
        normalize: default True.
        trim: default True. trim all silence that is longer than 0.2s and louder than 1% volume
        phone_call: default False.
    Raises:
        IOError: If the directory does ot exists or the directory did not contain any wav files.
    """

    def __init__(
        self,
        data,
        clip_frames=125,
        **kwargs,
    ) -> None:
        super().__init__(data, **kwargs)
        self.clip_frames = clip_frames

    def read_and_tile(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data.iloc[index]
        total_frames = item["video_frames"]

        # video frames less than wanted clip frames
        video = self.read_video(index)  # video (T, H, W, C)
        num_to_tile = math.ceil(self.clip_frames / total_frames)
        video = torch.tile(video, (num_to_tile, 1, 1, 1, 1))[: self.clip_frames, ...]
        waveform = self.read_audio(index)  # audio (C, L)
        audio = torch.tile(waveform, num_to_tile)[
            :, : int(waveform.shape[-1] * (self.clip_frames / total_frames))
        ]
        print("AudioVisualRandomClipDataset", waveform.shape, audio.shape, video.shape)
        return video, audio

    def get_start_frame(self, total_frames, clip_frames):
        start_frame = np.random.randint(0, total_frames - self.clip_frames)
        return start_frame

    def read_and_clip(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data.iloc[index]
        total_frames = item["video_frames"]

        # video frames is more than or equal wanted clip frames
        start_frame = self.get_start_frame(total_frames, self.clip_frames)
        end_frame = start_frame + self.clip_frames - 1

        if self.has_extracted_video_frames:
            video_img_path = item["video_img_path"]
            video = read_video_clip_from_frames(video_img_path, start_frame, end_frame)
        else:
            video_path = item["video_path"]
            video = read_random_video_clip_cv2(
                video_path, start_frame=start_frame, end_frame=end_frame
            )
        start_sample = int(start_frame / item["video_fps"] * item["audio_fps"])
        end_sample = start_sample + int(
            self.clip_frames / item["video_fps"] * item["audio_fps"]
        )
        waveform = self.read_audio(index)  # audio (C, L)
        audio = waveform[:, start_sample:end_sample]

        if waveform.shape[-1] < end_sample:
            audio = torch.nn.functional.pad(audio, (0, end_sample - waveform.shape[-1]))

        # print(start_sample, end_sample, waveform.shape, audio.shape, video.shape)
        return video, audio

    def read_clip(self, index: int) -> dict:
        item = self.data.iloc[index]
        total_frames = item["video_frames"]

        # video frames less than wanted clip frames
        if total_frames < self.clip_frames:
            video, audio = self.read_and_tile(index)
        else:
            # video frames is more than or equal wanted clip frames
            video, audio = self.read_and_clip(index)
        return video, audio

    @deal_possible_error
    def __getitem__(self, index: int) -> dict:
        res = self.read_metadata(index)
        res["video"], res["audio"] = self.read_clip(index)
        res = self.apply_transforms(res)
        return res


# -

# ### Audio-Visual Middle Clip Dataset

# + editable=true slideshow={"slide_type": ""}
class AudioVisualMiddleClipDataset(AudioVisualRandomClipDataset):
    """Torch dataset to load data from a provided directory.

    Args:
        data: A pandas dataframe, whose 'path' column is the path for each audio file.
        sample_rate: The used sample rate for the audio
        amount: default None. If not none, it means the number of used audio
        normalize: default True.
        trim: default True. trim all silence that is longer than 0.2s and louder than 1% volume
        phone_call: default False.
    Raises:
        IOError: If the directory does ot exists or the directory did not contain any wav files.
    """

    def get_start_frame(self, total_frames, clip_frames):
        start_frame = (total_frames - self.clip_frames) // 2
        return start_frame
# -


