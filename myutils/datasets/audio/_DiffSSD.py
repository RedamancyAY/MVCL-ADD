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
# %%
from myutils.datasets.base import AudioDataset
from myutils.tools import read_file_paths_from_folder, to_list

# %%
ROOT_PATH = "/home/ay/data2/DiffSSD/synthetic_speech_dataset_zenodo_v1"

# %%
METHODS = [
    "real",
    "xttsv2",
    "playht",
    "yourtts",
    "gradtts",
    "wavegrad2",
    "prodiff",
    "openvoicev2",
    "diffgantts",
    "elevenlabs",
    "unitspeech",
]


# %%
class DiffSSD_AudioDs(AudioDataset):

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
        data = pd.read_csv(os.path.join(root_path, "metadata.csv"))
        
        data["relative_path"] = data["dst_path"]
        data["label"] = data["target"].apply(lambda x: 1 if x == 0 else 0)
        data['split'] = data['set']


        data["vocoder_method"] = data['method_name'].apply(lambda x: 'real' if x in ['librispeech', 'ljspeech'] else x)
        data["vocoder_label"] = data["vocoder_method"].apply(lambda x: METHODS.index(x))

        self.update_audio_path(data, root_path)
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
        for split in ["train", "val", "test"]:
            _data = data.query(f'split == "{split}"').reset_index(drop=True)
            sub_datas.append(_data)
        
        return Namespace(
            train=sub_datas[0],
            val=sub_datas[1],
            test=sub_datas[2],
        )


# %%
# ds = DiffSSD_AudioDs(root_path=ROOT_PATH, )
# data = ds.data
# splits = ds.get_splits()

# %% [markdown]
# check split and methods:
#
#
# ```python
# vocoder_method split label method_name          
# diffgantts     test  0     diffgantts       5000
# elevenlabs     test  0     elevenlabs       2500
#                train 0     elevenlabs       2000
#                val   0     elevenlabs        500
# gradtts        test  0     gradtts          2500
#                train 0     gradtts          2000
#                val   0     gradtts           500
# openvoicev2    test  0     openvoicev2     12500
#                train 0     openvoicev2     10000
#                val   0     openvoicev2      2500
# playht         test  0     playht           5000
# prodiff        test  0     prodiff          2500
#                train 0     prodiff          2000
#                val   0     prodiff           500
# real           test  1     librispeech      5563
#                            ljspeech         6550
#                train 1     librispeech      4450
#                            ljspeech         5240
#                val   1     librispeech      1113
#                            ljspeech         1310
# unitspeech     test  0     unitspeech       5000
# wavegrad2      test  0     wavegrad2        2500
#                train 0     wavegrad2        2000
#                val   0     wavegrad2         500
# xttsv2         test  0     xttsv2           2500
#                train 0     xttsv2           2000
#                val   0     xttsv2            500
# yourtts        test  0     yourtts          2500
#                train 0     yourtts          2000
#                val   0     yourtts           500
# ```
