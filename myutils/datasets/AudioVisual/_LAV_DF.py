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
import os
import random
from argparse import Namespace
from enum import Enum
from typing import NamedTuple, Union

import numpy as np
import pandas as pd

from myutils.tools import check_dir, read_file_paths_from_folder, to_list
# -

from ..base import AudioVisualDataset


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# from base import AudioVisualDataset
# -

# | | Train | Val | Test|
# |-|-|-|-|
# |0|38178 | 15410 | 12742|
# |1|40525|16091| 13358|
# |Total| 78703| 31501|26100|

class LAV_DF(AudioVisualDataset):
    def __init__(self, root_path):
        super().__init__(root_path)
        self.data = self.read_metadata(root_path)

        self.data = self.set_video_audio_paths(self.data)

    def set_video_audio_paths(self, data):
        data["video_path"] = data["file"].apply(
            lambda x: os.path.join(self.root_path, x)
        )
        data["audio_path"] = data["file"].apply(
            lambda x: os.path.join(self.root_path, x)
        )
        return data

    def _set_video_metadatas(self, data):
        data["video_label"] = data["modify_video"].apply(lambda x: 1 if x else 0)
        data["video_periods"] = data.apply(
            lambda x: [] if not x["modify_video"] else x["fake_periods"], axis=1
        )
        data["video_n_fakes"] = data["video_periods"].apply(lambda x: len(x))
        data = self.read_video_info(data)

        return data

    def _set_audio_metadatas(self, data):
        data["audio_label"] = data["modify_audio"].apply(lambda x: 1 if x else 0)
        data["audio_periods"] = data.apply(
            lambda x: [] if not x["modify_audio"] else x["fake_periods"], axis=1
        )
        data["audio_n_fakes"] = data["audio_periods"].apply(lambda x: len(x))
        data = self.read_audio_info(data)
        return data

    def _read_metadata(self, root_path, data_path):
        data = pd.read_json(os.path.join(self.root_path, "metadata.min.json"))
        data["split"] = data["split"].replace("dev", "val")
        data = self.set_video_audio_paths(data)

        data = self._set_audio_metadatas(data)
        data = self._set_video_metadatas(data)

        return data

    def split_data(self, *args, **kwargs):
        datasets = {}
        for _split in ["train", "val", "test"]:
            _data = self.data.query(f"split == '{_split}'")
            datasets[_split] = _data

        return Namespace(**datasets)

# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# lav_df = LAV_DF(root_path="/home/ay/data/0-原始数据集/LAV-DF")
