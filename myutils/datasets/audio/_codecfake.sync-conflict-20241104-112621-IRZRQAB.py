# -*- coding: utf-8 -*-
# %% [markdown]
# - Github:[xieyuankun/Codecfake: This is the official repo of our work titled "The Codecfake Dataset and Countermeasures for the Universally Detection of Deepfake Audio".](https://github.com/xieyuankun/Codecfake)

# %%
import os
import random
import re
from argparse import Namespace
from enum import Enum
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from pandarallel import pandarallel
from tqdm.auto import tqdm

# %%
from myutils.datasets.base import AudioDataset
from myutils.tools import read_file_paths_from_folder, to_list

# %% [markdown]
# # Download dataset

# %% [markdown]
# 1. Download the zip file from Github: [xieyuankun/Codecfake](https://github.com/xieyuankun/Codecfake)
# 2. After uncompressing the 'zip' files, please rearrage the folder as:
# ```
# ├── Codecfake
# │   ├── label
# │   │   └── *.txt
# │   ├── train
# │   │   └── *.wav (740,747 samples)
# │   ├── val
# │   │   └── *.wav (92,596 samples)
# │   ├── test
# │   │   └── C1
# │   │   │   └── *.wav (26,456 samples)
# │   │   └── C2
# │   │   │   └── *.wav (26,456 samples)
# │   │   └── C3
# │   │   │   └── *.wav (26,456 samples)
# │   │   └── C4
# │   │   │   └── *.wav (26,456 samples)
# │   │   └── C5
# │   │   │   └── *.wav (26,456 samples)
# │   │   └── C6
# │   │   │   └── *.wav (26,456 samples)
# │   │   └── C7
# │   │   │   └── *.wav (145,505 samples)
# │   │   └── A1
# │   │   │   └── *.wav (8,902 samples）
# │   │   └── A2
# │   │   │   └── *.wav (8,902 samples）
# │   │   └── A3
# │   │   │   └── *.wav (99,112 samples）
# ```

# %% [markdown]
# > In the original paper, the author said: **"Given that some of the original Audiocaps audio links are no longer active, we use the currently available `49,274` audio samples as the real source domain for A3."**
#
# However, for the A3 test subset, the real samples are not provided. Therefore, in A3, there are only 49838 samples. Please download the Audiocaps dataset, then convert its samples into 16k HZ,  and then push all the audios in the train subset of Audiocaps into `Codecfake/test/A3/`.

# %%
ROOT_PATH = "/mnt/data1/zky/Codecfake16k"

# %% [markdown]
# # Generate metadata from the original label files

# %% [markdown]
# ![|600](https://raw.githubusercontent.com/RedamancyAY/CloudImage/main/img202409271618092.png)

# %% [markdown]
# In the dataset, the used synthesizers are `["C1", "C2", "C3", "C4", "C5", "C6", "C7", "L1", "L2", "L3"]`. We first **read all the txt files** in the `Codecfake16k/label/` folder, and then generate a pandas DataFrame. The generated metadata is a csv file, that is:

# %% [markdown]
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>filename</th>
#       <th>label</th>
#       <th>method</th>
#       <th>relative_path</th>
#       <th>split</th>
#       <th>vocoder</th>
#       <th>vocoder_label</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>1129288</th>
#       <td>F06_p313_116.wav</td>
#       <td>0</td>
#       <td>6</td>
#       <td>train/F06_p313_116.wav</td>
#       <td>train</td>
#       <td>C6</td>
#       <td>6</td>
#     </tr>
#     <tr>
#       <th>623492</th>
#       <td>F02_p276_032.wav</td>
#       <td>0</td>
#       <td>2</td>
#       <td>train/F02_p276_032.wav</td>
#       <td>train</td>
#       <td>C2</td>
#       <td>2</td>
#     </tr>
#     <tr>
#       <th>1080750</th>
#       <td>F06_SSB13920324.wav</td>
#       <td>0</td>
#       <td>6</td>
#       <td>train/F06_SSB13920324.wav</td>
#       <td>train</td>
#       <td>C6</td>
#       <td>6</td>
#     </tr>
#     <tr>
#       <th>327989</th>
#       <td>F07_p308_343.wav</td>
#       <td>0</td>
#       <td>7</td>
#       <td>test/C7/F07_p308_343.wav</td>
#       <td>test-C7</td>
#       <td>C7</td>
#       <td>7</td>
#     </tr>
#     <tr>
#       <th>1159904</th>
#       <td>F04_SSB19350072.wav</td>
#       <td>0</td>
#       <td>4</td>
#       <td>train/F04_SSB19350072.wav</td>
#       <td>train</td>
#       <td>C4</td>
#       <td>4</td>
#     </tr>
#   </tbody>
# </table>

# %%
METHODS = ["real", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "L1", "L2", "L3"]


# %%
def generate_metadata(root_path):

    save_path = os.path.join(root_path, 'metadata.csv')
    if os.path.exists(save_path):
        return pd.read_csv(save_path)
    
    datas = []
    for file in read_file_paths_from_folder(root_path, exts="txt"):
        if ".ipynb_checkpoints" in file:
            continue
        _data = pd.read_csv(file, sep=" ", names=["filename", "label", "method"])
    
        #### obtain relative path and split in the Codecfake folder
        txt_filename = os.path.split(file)[1].split(".")[0]  ### train, val, A1, ..., C1, C2, ....
        _data["relative_path"] = _data["filename"].map(
            lambda x: f"{'' if txt_filename in ['train', 'val'] else 'test/'}{txt_filename}/{x}"
        )
        _data["split"] = _data["filename"].map(
            lambda x: txt_filename if txt_filename in ['train', 'val'] else f'test-{txt_filename}'
        )
        
        #### process labels and vocoder methods
        _data["label"] = _data["label"].map(lambda x: 1 if x == "real" else 0)
        _data["vocoder"] = _data.apply(
            lambda x: "real"
            if x["label"] == 1
            else ("C" + str(x["method"]) if str(x["method"]) in ["1", "2", "3", "4", "5", "6", "7"] else x["method"]),
            axis=1,
        )
        _data["vocoder_label"] = _data['vocoder'].map(lambda x: METHODS.index(x))
        datas.append(_data)
    
    _data = pd.concat(datas, ignore_index=True)

    _data.to_csv(save_path, index=False)
    return _data


# %%
# _data = generate_metadata(ROOT_PATH)
# print(len(_data))
# _data.groupby(['split', 'vocoder']).count()

# %% [markdown]
# # Read wav files and obtain corresponding labels

# %%
def read_wav_paths(root_path):
    """
    Reads WAV file paths from a given root directory and returns a DataFrame with file information.

    Args:
        root_path (str): The root directory path to search for WAV files.

    Returns:
        pd.DataFrame: A DataFrame containing columns:
            - audio_path: Full path to the WAV file.
            - relative_path: Path relative to the root directory.
            - filename: Name of the WAV file.

    Note:
        This function uses the `read_file_paths_from_folder` function to retrieve file paths.
    """
    filepaths = read_file_paths_from_folder(root_path, exts=["wav"])
    
    data = pd.DataFrame(filepaths, columns=["audio_path"])
    
    data["relative_path"] = data["audio_path"].map(lambda x: x.split(f"{root_path}/")[1])
    data["filename"] = data["audio_path"].map(lambda x: os.path.split(x)[1])
    return data


# %%
# data = read_wav_paths(ROOT_PATH)
# data

# %% [markdown]
# after reading all the wav files, we find that there are only **1205196** wav files, smaller the number "**1254470**" in labels, since the 49,274 real samples in A3 are not provided.

# %%
class Codecfake_AudioDs(AudioDataset):

    def update_audio_path(self, data:pd.DataFrame, root_path: str):
        """
        Update the audio path in the given data.

        Args:
            data (pd.DataFrame): The DataFrame containing the audio data.
            root_path (str): The root path to prepend to the relative paths.

        Returns:
            None
        """
        data["audio_path"] = data["relative_path"].apply(
            lambda x: os.path.join(root_path, x)
        )
    
    def postprocess(self):
        """
        Perform post-processing on the dataset.

        Updates the audio paths and sets the vocoders.

        Returns:
            None
        """
        self.update_audio_path(self.data, self.root_path)
        self.vocoders = METHODS
    
    def _read_metadata(self, root_path, *args, **kwargs):
        """
        Read metadata for the audio dataset.

        Args:
            root_path (str): The root path of the audio files.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pd.DataFrame: The processed metadata.
        """
        _realdata = read_wav_paths(root_path)
        _metadata = generate_metadata(root_path)
        
        data = pd.merge(_realdata, _metadata, on='relative_path')
        self.update_audio_path(data, root_path)
        data = self.read_audio_info(data)  # read fps and length
        return data

    def get_splits(self):
        """
        Get train/val/test splits according to the public splits.

        Returns:
            Namespace: An object containing train, val, and test data splits.
                train (pd.DataFrame): Training data.
                val (pd.DataFrame): Validation data.
        """

        data = self.data

        sub_datas = []
        for split in ["train", "val"]:
            _data = data.query(f'split == "{split}"').reset_index(drop=True)
            sub_datas.append(_data)

        test_datas = []
        for method in METHODS[1:]:
            if method.startswith('L'):
                method = 'A' + method[1]
            item = 'test-' + method
            _data = data.query(f'split == "{item}"').reset_index(drop=True)
            test_datas.append(_data)
        
        return Namespace(
            train=sub_datas[0],
            val=sub_datas[1],
            test=test_datas,
        )


# %%
ds = Codecfake_AudioDs(root_path=ROOT_PATH)
data = ds.data
splits = ds.get_splits()

# %%
# splits.test[9]
