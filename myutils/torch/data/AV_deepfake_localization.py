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
# -

from .AudioVisual import AudioVisualDataset, deal_possible_error
from .utils import get_gt_iou_map


# + tags=["style-solution", "active-ipynb"]
# from AudioVisual import AudioVisualDataset, deal_possible_error
# from utils import get_gt_iou_map
# -

def get_frame_label_from_fake_periods(n_frames, fake_periods, fps, padding=0):
    """get frame label from the fake periods

    Args:
        n_frames: the number of total frames
        fake_periods: the fake periods, for example, [[0.1, 1.2], [5.2, 7.2]]
        fps: frames per second
        padding: the n_frames < padding, it will be padding.

    Returns:
        a tensor denoteing the frame_label that with length of max(n_frames, padding)

    """
    frame_label = torch.ones(max(n_frames, padding))
    for _fake_periods in eval(fake_periods):
        start_frame = int(_fake_periods[0] * fps)
        end_frame = int(_fake_periods[1] * fps)
        frame_label[start_frame : end_frame + 1] = 0
    return frame_label


from hashlib import md5, sha256


def hash_args(*args):
    s = ""
    for x in args:
        s += str(x)
    _md5 = md5()
    _sha = sha256()
    _md5.update(str(s).encode("utf-8"))
    _sha.update(str(s).encode("utf-8"))
    res = _md5.hexdigest() + _sha.hexdigest()
    return res


# + editable=true slideshow={"slide_type": ""}
class AV_Deepfake_Localization_Dataset(AudioVisualDataset):
    @deal_possible_error
    def load_gt_iou_map(self, fake_periods, frames, max_duration=40, padding=512):
        """generate gt_iou_map for video or audio

        Args:
            item: a row of a pd.Dataframe, actually a dict.
            modality: must be 'video' or 'audio'

        Returns:
            the gt_iou_map.

        """
        fake_periods = eval(fake_periods)
        if not fake_periods:
            return torch.zeros(max_duration, padding)

        path_gt_iou_map = "/home/ay/data/3-middle-res/0-gt-iou-map/{}.pt".format(
            hash_args(fake_periods, frames, max_duration, padding)
        )
        if os.path.exists(path_gt_iou_map):
            x = torch.load(path_gt_iou_map).to_dense()
            # print('load gt_iou_map from, ', path_gt_iou_map)
            return x

        # print(fake_periods, frames)
        gt_iou_map = get_gt_iou_map(
            frames=frames,
            video_labels=fake_periods,
            temporal_scale=frames,
            fps=25,
            max_duration=max_duration,
            padding=padding,
        )
        torch.save(gt_iou_map.to_sparse(), path_gt_iou_map)
        return gt_iou_map

    @deal_possible_error
    def read_metadata(self, index: int) -> dict:
        """read the metadata of the `index`-th item

        Args:
            index: the row index of self.data (a pd.Dataframe)

        Returns:
            a dict that contains the metadata of the item.
        """

        item = self.data.iloc[index]

        res = {
            "audio_fps": self.audio_fps,
            "video_fps": item["video_fps"],
            "video_path": item["video_path"],
            "audio_path": item["audio_path"],
            "video_label": item["video_label"],
            "audio_label": item["audio_label"],
            "video_frames": item["video_frames"],
            "audio_frames": item["video_frames"],
            "video_frame_label": get_frame_label_from_fake_periods(
                n_frames=item["video_frames"],
                fake_periods=item["video_periods"],
                fps=item["video_fps"],
            ),
            "audio_frame_label": get_frame_label_from_fake_periods(
                n_frames=item["video_frames"],
                fake_periods=item["audio_periods"],
                fps=25,  #  regrad 40ms audio clip as a frame, thus the fps is 25
            ),
        }

        # get gt_iou_map for video and audio modalities.
        max_duration = 40
        padding = 512
        if not item["fake_periods"]:  # fake periods is []
            res["av_gt_iou_map"] = res["video_gt_iou_map"] = res[
                "audio_gt_iou_map"
            ] = torch.zeros(max_duration, padding)
        else:
            res["av_gt_iou_map"] = self.load_gt_iou_map(
                item["fake_periods"], item["video_frames"], max_duration=max_duration, padding=padding
            )
            for _m in ["video", "audio"]:
                if item[f"{_m}_periods"] == item["fake_periods"]:
                    res[f"{_m}_gt_iou_map"] = res["av_gt_iou_map"]
                else:
                    res[f"{_m}_gt_iou_map"] = self.load_gt_iou_map(
                        item["video_periods"],
                        item["video_frames"],
                        max_duration=max_duration,
                        padding=padding,
                    )

        if self.has_extracted_video_frames:
            res["video_img_path"] = item["video_img_path"]
        return res

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# from myutils.datasets.AudioVisual import LAV_DF
#
# lavdf = LAV_DF(root_path="/home/ay/data/0-原始数据集/LAV-DF")
#
# item = lavdf.data.iloc[2]
