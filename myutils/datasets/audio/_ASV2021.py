# +
import os
import random
import re
from argparse import Namespace
from enum import Enum
from typing import NamedTuple, Union
from IPython.display import HTML, display

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm
# -

from myutils.datasets.base import AudioDataset
from myutils.tools import read_file_paths_from_folder, to_list

VOCODERs = [
    "bonafide",
    "neural_vocoder_autoregressive",
    "neural_vocoder_nonautoregressive",
    "traditional_vocoder",
    "unknown",
    "waveform_concatenation",
]

COMPRESSIONs = ['high_m4a',
 'high_mp3',
 'high_ogg',
 'low_m4a',
 'low_mp3',
 'low_ogg',
 'mp3m4a',
 'nocodec',
 'oggm4a']


class ASV2021_AudioDs(AudioDataset):
    def postprocess(self):
        self.data["audio_path"] = self.data["file"].apply(
            lambda x: os.path.join(self.root_path, x)
        )
        self.data["compression_label"] = self.data["compression"].apply(lambda x: COMPRESSIONs.index(x))
        
        self.vocoders = VOCODERs

    def _read_metadata(self, root_path, *args, **kwargs):
        data = pd.read_csv(os.path.join(root_path, "used_metadata.csv"))
        # data['file'] = data['trial'].apply(lambda x: f"flac/{x}.flac")
        data["file"] = data["trial"].apply(lambda x: f"wav/{x}.wav")
        data["label"] = data["label"].apply(lambda x: 1 if x == "bonafide" else 0)
        data["vocoder_label"] = data["vocoder"].apply(lambda x: VOCODERs.index(x))

        data["audio_path"] = data["file"].apply(lambda x: os.path.join(root_path, x))

        data["split"] = data["subset"].apply(
            lambda x: "test" if x == "eval" else "train"
        )

        
        val_data = data.query("subset == 'progress'").sample(10000, random_state=42)
        data.loc[val_data.index, "split"] = "val"

        data = self.read_audio_info(data)  # read fps and length
        return data

    def get_splits(self):
        """
            Get train/val/test splits according to the language.

        Args:

        Returns:
            Namespace(train, val, test)
        """

        data = self.data

        sub_datas = []
        for split in ["train", "val", "test"]:
            _data = data.query(f'split == "{split}"').reset_index(drop=True)
            sub_datas.append(_data)

        data = sub_datas[-1]
        real_data = data.query(
                f"vocoder == '{VOCODERs[0]}'"
            ).reset_index(drop=True)
        inner_data = data.query(
            "attack in ['A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']"
        ).reset_index(drop=True)
        other_index = [x for x in data.index if x not in inner_data.index]
        data2 = self.get_test_splits(data.loc[other_index])

        inner_data = pd.concat([inner_data, real_data], ignore_index=True)
        data2.insert(0, inner_data)

        return Namespace(
            train=sub_datas[0],
            test=data2,
            val=sub_datas[1],
        )

    def get_whole_test_split(self):
        data = self.data
        _data = data.query(f'split == "test"').reset_index(drop=True)
        return _data
    
    def get_test_splits(self, data=None):
        """
            Get train/val/test splits according to the language.

        Args:

        Returns:
            Namespace(train, val, test)
        """

        data = data if data is not None else self.data.query("split == 'test'")

        sub_datas = []
        for vocoder in VOCODERs[1:]:
            _data = data.query(
                f"vocoder == '{vocoder}' or vocoder == '{VOCODERs[0]}'"
            ).reset_index(drop=True)
            sub_datas.append(_data)

        return sub_datas

# root_path = '/home/ay/data/ASVspoof2021_DF_eval'
# ds = ASV2021_AudioDs(root_path=root_path)
# data = ds.data

# +
## get all the compression settings
# set(list(data['compression']))
# -

# splits = ds.get_splits()

# <center><img src="https://raw.githubusercontent.com/RedamancyAY/CloudImage/main/img202408111656939.png" width="400"/></center>
#
#
# - train and val: A07-A19
# - test: A07-A19, HUB-B00-HUB-B01, HUB-D01-HUB-D05, HUB-N03-HUB-N20, SPO-N03-SPO-N18, Task1-team01-Task1-team33, Task2-team01-Task2-team33

# +
# pd.set_option("display.max_rows", 200, "display.max_columns", 20)
# data.groupby(['split', 'label']).count()

# +
# data.query("label == 1").groupby(['compression', 'split']).count()

# +
# from IPython.display import HTML, display

# display(HTML(splits.test[0].groupby(["vocoder", "attack", "subset"]).count().to_html()))
# -


