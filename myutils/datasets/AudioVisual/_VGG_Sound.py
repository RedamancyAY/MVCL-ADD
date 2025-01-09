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
import re

import pandas as pd
import torchvision
from tqdm import tqdm

from myutils.tools import check_dir, read_file_paths_from_folder, to_list
from myutils.tools.audio import AudioConverter
from myutils.tools.video import VideoConverter

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# # import sys
# # cur_path = os.path.abspath(__file__)
# # cur_dir = os.path.split(cur_path)
# # print(cur_dir)

# + editable=true slideshow={"slide_type": ""}
from ..base import AudioVisualDataset


# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# from myutils.datasets.base import AudioVisualDataset
# -

class VGGSound_PrepareDataset:

    """
    Extract the audio stream and the visual stream from the downloaded videos.
    the new folder will be:
    VGG-Sound
        - audio
        - video
        - vggsound.csv
    """

    def __init__(self, source_folder, dest_folder):
        """
        source_folder: the folder that you download and unzip the dataset
        dest_folder: the destination folder that save the audio and visual streams.
        """

        video_paths = read_file_paths_from_folder(source_folder, exts=["mp4"])
        if len(video_paths) != 199176:
            print(
                f"Warning!!! VGGSound should have 199176 video, but your folder contains {len(video_paths)}"
            )

        self.data = pd.DataFrame(video_paths, columns=["path"])
        self.source_folder = source_folder
        self.dest_folder = dest_folder

    def convert_audio(self, path):
        id = os.path.basename(path).split(".mp4")[0]
        new_name = id + ".wav"
        new_path = os.path.join(self.dest_folder, "audio", new_name)
        if not os.path.exists(new_path):
            self.audio_converter.convert(path, new_path)

    def convert_video(self, path):
        filename = os.path.basename(path)
        new_path = os.path.join(self.dest_folder, "video", filename)
        self.video_converter.convert(path, new_path)

    def convert_all_audios(self):
        """
        By default, the extracted audio is of 1 channel and 16000 sampling rate
        """
        from pandarallel import pandarallel

        pandarallel.initialize(progress_bar=True, nb_workers=50)
        self.audio_converter = AudioConverter()
        data = self.data.sample(frac=1).reset_index(drop=True)
        data.parallel_apply(lambda x: self.convert_audio(x["path"]), axis=1)

    def convert_all_videos(self, fps=25):
        """
        By default, the extracted video is of 25 fps.
        """
        from pandarallel import pandarallel

        pandarallel.initialize(progress_bar=True, nb_workers=5)
        self.video_converter = VideoConverter(fps=fps)
        data = self.data.sample(frac=1).reset_index(drop=True)
        data.parallel_apply(lambda x: self.convert_video(x["path"]), axis=1)


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# source_folder = "/home/ay/data/0-原始数据集/VGGSound"
# dest_folder = "/home/ay/data/DATA/dataset/2-audiovisual/VGG-Sound"
#
# P = VGGSound_PrepareDataset(source_folder=source_folder, dest_folder=dest_folder)
# # P.convert_all_audios()
# P.convert_all_videos(fps=1)

# + editable=true slideshow={"slide_type": ""}
class VGGSound(AudioVisualDataset):
    """
    Generate metadata for the VGGSound datasets.
    """

    def __init__(self, root_path="/home/ay/data/DATA/dataset/2-audiovisual/VGG-Sound"):
        """
        Args:
            root_path: the path of VGGSound dataset. Note that the root path must contain "/VGG-Sound/"
        """

        self.root_path = root_path if not root_path.endswith("/") else root_path[:-2]
        self.data = self.read_metadata(self.root_path)


    def read_org_metadata(self, root_path):
        data = pd.read_csv(
            os.path.join(root_path, "vggsound.csv"),
            header=None,
            names=["video_id", "start_sec", "desc", "split"],
        )
        
        # get label
        all_desc = sorted(list(set(data["desc"])))
        data["label"] = data["desc"].apply(lambda x: all_desc.index(x))
        data["filename"] = data.apply(
            lambda x: f"{x['video_id']}_{'%06d'%x['start_sec']}", axis=1
        )
        return data
    
    def _read_metadata(self, root_path, data_path):
        """
        read all the metadatas of audio files from the root_path
        """

        data = self.read_org_metadata(root_path)

        # get video path and audio path
        data["video_path"] = data["filename"].apply(
            lambda x: os.path.join(root_path, "video", x + ".mp4")
        )
        data["video_img_path"] = data["filename"].apply(
            lambda x: os.path.join(root_path, "imgs", x)
        )
        data["audio_path"] = data["filename"].apply(
            lambda x: os.path.join(root_path, "audio", x + ".wav")
        )
        data["is_video_exists"] = data["video_path"].apply(lambda x: os.path.exists(x))
        data["is_audio_exists"] = data["audio_path"].apply(lambda x: os.path.exists(x))
        print(len(data), data['video_path'][0])

        # use videos that have video and audio
        data = data.query("is_video_exists == 1 & is_audio_exists == 1")

        print(len(data))
        
        data = self.read_video_info(data)
        data = self.read_audio_info(data)

        # save metadata to file
        data.to_csv(data_path, index=False)
        return data

# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# x = VGGSound(root_path="/home/ay/data/DATA/dataset/2-audiovisual/VGG-Sound")
# -

# # Vgg-Sound

# ## 原始metadata

# 可以看出，不同的situation共有309种，根据situation来创建label：0-308。

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-student"]
# data2 = pd.read_csv(
#     "/home/ay/data/DATA/dataset/0-audio/VGG-Sound/vggsound.csv",
#     names=["id", "start_second", "situation", "split"],
# )
# data2.groupby("situation").count()
