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
# After uncompressing the 'zip' files, please rearrage the folder as:
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
# > Given that some of the original Audiocaps audio links are no longer active, we use the currently available 49,274 audio samples as the real source domain for A3.
#
# For the A3 test subset, the real samples are not provided.

# %%
ROOT_PATH = "/mnt/data1/zky/Codecfake16k"

# %% [markdown]
# # Generate metadata from the original label files

# %% [markdown]
# ![|600](https://raw.githubusercontent.com/RedamancyAY/CloudImage/main/img202409271618092.png)

# %% [markdown]
# The authors put the labels in the `label/*.txt` files, and we use the following script to generate the metadata.
#
# The used synthesizers are `["C1", "C2", "C3", "C4", "C5", "C6", "C7", "A1", "A2", "A3"]`. However, in the label text files, the "A1, A2, A3" were wrote as "L1, L2, L3' Therefore we need to rename them.
#
# :::warning
#
# Note, in the A1.txt, A2.txt, A3.txt, the `method` column for real samples are assigned to L1, L2 and L3 respectively. This is wrong!!!!!
#
# :::
#

# %%
METHODS = ["real", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "A1", "A2", "A3"]


# %%
def read_txt_labels(root_path):
    """Reads label text files and returns a concatenated dataframe.

    This function reads all `.txt` files in the "label" directory within the specified `root_path`.

    Args:
        root_path (str): The path to the root directory of Codecfake, containing the "label" folder.

    Returns:
        DataFrame: A pandas DataFrame containing all the label data from the text files.

    Raises:
        FileNotFoundError: If the "label" directory does not exist within `root_path`.
        ValueError: If no `.txt` files are found in the "label" directory.

    Example:
        >>> read_txt_labels('/path/to/root')
        DataFrame with columns ['filename', 'label', 'method']
    """

    label_folder = os.path.join(
        root_path, "label"
    )  # Construct the path to the label folder

    # Check if the label folder exists
    if not os.path.exists(label_folder):
        raise FileNotFoundError(f"The directory {label_folder} does not exist.")

    # List all files in the label folder ending with .txt
    label_files = [file for file in os.listdir(label_folder) if file.endswith(".txt")]

    # Check if there are any .txt files
    if not label_files:
        raise ValueError("No .txt files found in the label directory.")

    data = []  # Initialize an empty list to store individual DataFrames

    # Iterate over each text file and read data
    for file in label_files:
        file_path = os.path.join(
            label_folder, file
        )  # Construct full path to the current file

        # Read the TXT file using pandas with specified separator and column names
        _data = pd.read_csv(
            file_path, sep=" ", names=["filename", "label", "method"], dtype=str
        )

        # Map method identifiers to descriptive names:
        ## `["C1", "C2", "C3", "C4", "C5", "C6", "C7", "A1", "A2", "A3"]`
        _data["vocoder_method"] = _data["method"].map(
            {
                "0": "real",
                "1": "C1",
                "2": "C2",
                "3": "C3",
                "4": "C4",
                "5": "C5",
                "6": "C6",
                "7": "C7",
                "L1": "A1",
                "L2": "A2",
                "L3": "A3",
            }
        )
        _data["vocoder_method"] = _data.apply(
            lambda x: "real" if x["label"] == "real" else x["vocoder_method"], axis=1
        )
        _data["vocoder_label"] = _data["vocoder_method"].apply(
            lambda x: METHODS.index(x)
        )
        _data["label"] = _data["label"].map(lambda x: 1 if x == "real" else 0)

        # Append the current DataFrame to the list
        data.append(_data)

    # Concatenate all individual DataFrames into a single DataFrame and reset the index
    data = pd.concat(data, ignore_index=True)

    return data


# %%
# data = read_txt_labels(ROOT_PATH)
# data.sample(100)

# %%
# from myutils.tools.pandas import check_same_labels_for_duplicated_column
# check_same_labels_for_duplicated_column(data, column1='filename', column2='label')

# %% [markdown]
# run this command to randomly output 5 rows:
# ```python
# print(data.sample(5).to_markdown())
# ```
# The generated metadata is a csv file, that is:

# %% [markdown]
# |         | filename            |   label | method   | vocoder_method   |   vocoder_label |
# |--------:|:--------------------|--------:|:---------|:-----------------|----------------:|
# |  133541 | F01_SSB19390107.wav |       0 | 1        | C1               |               1 |
# |  102298 | YCsop4JzOMZ0.wav    |       1 | L3       | real             |               0 |
# |  299159 | F07_SSB18090341.wav |       0 | 7        | C7               |               7 |
# | 1027611 | F06_SSB09660328.wav |       0 | 6        | C6               |               6 |
# |  522169 | F06_SSB18630104.wav |       0 | 6        | C6               |               6 |

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
    data["split"] = data["relative_path"].map(lambda x: x.split('/')[0])
    data["test_split"] = data["relative_path"].map(lambda x: x.split('/')[1] if len(x.split('/')) > 2 else 'null')
    return data


# %%
# data = read_wav_paths(ROOT_PATH)
# data
# # print(data.sample(5).to_markdown())

# %% [markdown]
# run this command to randomly output 5 samples
# ```python
# print(data.sample(5).to_markdown())
# ```

# %% [markdown]
# |         | audio_path                                              | relative_path               | filename            | split   | test_split   |
# |--------:|:--------------------------------------------------------|:----------------------------|:--------------------|:--------|:-------------|
# |  914874 | /mnt/data1/zky/Codecfake16k/train/F05_p225_147.wav      | train/F05_p225_147.wav      | F05_p225_147.wav    | train   | null         |
# |  801968 | /mnt/data1/zky/Codecfake16k/train/F04_SSB16860186.wav   | train/F04_SSB16860186.wav   | F04_SSB16860186.wav | train   | null         |
# |  699405 | /mnt/data1/zky/Codecfake16k/train/F03_SSB18370167.wav   | train/F03_SSB18370167.wav   | F03_SSB18370167.wav | train   | null         |
# | 1171954 | /mnt/data1/zky/Codecfake16k/val/F01_p258_054.wav        | val/F01_p258_054.wav        | F01_p258_054.wav    | val     | null         |
# |  337493 | /mnt/data1/zky/Codecfake16k/test/C7/F07_SSB11260234.wav | test/C7/F07_SSB11260234.wav | F07_SSB11260234.wav | test    | C7           |

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
        _metadata = read_txt_labels(root_path)
        
        
        data = pd.merge(_realdata, _metadata, on='filename')
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


        _test_data = data.query('split == "test"').reset_index(drop=True)
        test_datas = []
        for method in METHODS[1:]:
            _data = _test_data.query(f'test_split == "{method}"').reset_index(drop=True)
            test_datas.append(_data)
        
        return Namespace(
            train=sub_datas[0],
            val=sub_datas[1],
            test=test_datas,
        )

# %% [markdown]
# use the following  code to generate a metadata csv file.

# %%
# ds = Codecfake_AudioDs(root_path=ROOT_PATH)
# data = ds.data
# splits = ds.get_splits()

# %%
# splits.test[9]
