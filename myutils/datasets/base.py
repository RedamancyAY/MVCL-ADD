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
import os
from argparse import Namespace
from enum import Enum
from typing import Union

import pandas as pd

from myutils.tools.audio import read_audio_fps_len
from myutils.tools import check_dir, read_file_paths_from_folder, to_list
from myutils.tools.pandas import DF_spliter
from pandarallel import pandarallel
from tqdm.auto import tqdm


# -

# ---

class Base:
    def __init__(self, root_path, *args, **kwargs):
        """
        When crate a entry of WaveFake, it will read all the metadatas from the root_path

        Args:
            root_path: the path of WaveFake dataset. Note that the path must contain "/WaveFake/"
        """
        self.root_path = root_path if not root_path.endswith("/") else root_path[:-2]
        self.data = self.read_metadata(self.root_path)
        
        
        self.initial_property(root_path, *args, **kwargs)
        self.postprocess()

    
    def initial_property(self, root_path, *args, **kwargs):
        pass
    
        
    def postprocess(self):
        pass
    
        

    
    
    def save_metadata(self, data, data_path = None):
        if data_path is None:
            data_path = os.path.join(self.root_path, "dataset_info.csv")
        data.to_csv(data_path, index=False)
    
    def read_metadata(self, root_path=None, re_generate=False):
        """
        read all the metadatas of audio files from the root_path
        """
        if root_path is None:
            root_path = self.root_path
        data_path = os.path.join(root_path, "dataset_info.csv")
        if os.path.exists(data_path) and not re_generate:
            return pd.read_csv(data_path)
        else:
            data = self._read_metadata(root_path, data_path)
            self.save_metadata(data)
            return data

    def _read_metadata(self, root_path, data_path=None):
        raise NotImplementedError

    def split_data(self, data: pd.DataFrame = None, splits=[0.6, 0.2, 0.2], refer=None, return_list=False):
        if data is None:
            data = self.data

        if refer is None:
            sub_datas = DF_spliter.split_df(data, splits)
        else:
            sub_datas = DF_spliter.split_by_number_and_column(data, splits, refer=refer)
            

        if return_list:
            return sub_datas
        else:
            return Namespace(
                train=sub_datas[0],
                test=sub_datas[-1],
                val=None if len(splits) == 2 else sub_datas[1],
            )


class AudioDataset(Base):

    
    
    def read_fps_length(self, data: pd.DataFrame, column='audio_path') -> pd.DataFrame:
        pandarallel.initialize(progress_bar=True, nb_workers=15)
        data[["fps", "length"]] = data.parallel_apply(
            lambda x: tuple(read_audio_fps_len(x[column])), axis=1, result_type="expand"
        )
        return data


    def read_audio_info(self, data: pd.DataFrame) -> pd.DataFrame:
        from myutils.tools.audio.read_info import read_audio_fps_len

        pandarallel.initialize(progress_bar=True, nb_workers=20)
        data[
            ["audio_fps", "audio_len"]
        ] = data.parallel_apply(
            lambda x: tuple(read_audio_fps_len(x["audio_path"])),
            axis=1,
            result_type="expand",
        )
        return data


class AudioVisualDataset(AudioDataset):

    
    
    def read_video_info(self, data: pd.DataFrame) -> pd.DataFrame:
        from myutils.tools.video.read_info import read_video_fps_len_height_width

        pandarallel.initialize(progress_bar=True, nb_workers=10)
        data[
            ["video_fps", "video_frames", "video_height", "video_width"]
        ] = data.parallel_apply(
            lambda x: tuple(read_video_fps_len_height_width(x["video_path"])),
            axis=1,
            result_type="expand",
        )
        return data


class VideoDataset(Base):

    
    def read_video_info(self, data: pd.DataFrame) -> pd.DataFrame:
        from myutils.tools.video.read_info import read_video_fps_len_height_width

        pandarallel.initialize(progress_bar=True, nb_workers=10)
        data[
            ["video_fps", "video_frames", "video_height", "video_width"]
        ] = data.parallel_apply(
            lambda x: tuple(read_video_fps_len_height_width(x["video_path"])),
            axis=1,
            result_type="expand",
        )
        return data
