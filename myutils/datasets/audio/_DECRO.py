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
import random
from argparse import Namespace
from enum import Enum
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm
# -

from myutils.datasets.base import AudioDataset
from myutils.tools import read_file_paths_from_folder, to_list

# ---

# ## 数据集

# ### There are 63098 and 55283 utterances in the Chinese and English subsets, respectively. 
#
#
# <table>
#     <tr>
#   		 <td></td> 
#       	 <td colspan="3">English</td>   
#       	 <td colspan="3">Chinese</td>    
#     </tr>
#     <tr>
#   		 <td></td> 
#   		 <td>Train Set</td> 
#   		 <td>Dev Set</td> 
#   		 <td>Eval Set</td> 
#   		 <td>Train Set</td> 
#   		 <td>Dev Set</td> 
#   		 <td>Eval Set</td> 
#     </tr>
#     <tr>
#   		 <td>Bona-fide</td> 
#   		 <td>5129</td> 
#   		 <td>3049</td> 
#   		 <td>4306</td> 
#   		 <td>9000</td> 
#   		 <td>6109</td> 
#   		 <td>6109</td> 
#     </tr>
#     <tr>
#   		 <td>Spoofed</td> 
#   		 <td>17412</td> 
#   		 <td>10503</td> 
#   		 <td>14884</td> 
#   		 <td>17850</td> 
#   		 <td>12015</td> 
#   		 <td>12015</td> 
#     </tr>
#     <tr>
#   		 <td>Total</td> 
#   		 <td>22541</td>
#   		 <td>13552</td> 
#   		 <td>19190</td> 
#   		 <td>26850</td> 
#   		 <td>18124</td> 
#   		 <td>18124</td> 
#     </tr>
# </table>

VOCODERS = [
    "hifigan",
    "mbmelgan",
    "pwg",
    ["Tacotron", "tacotron"],
    ["fs2", "FastSpeech2", "fs2mtts"],
    ["starganv2", "StarGAN"],
    "vits",
    ["nvcnet", "nvcnet-cn"],
    ["baidu", "baidu_en"],
    "xunfei",
]


class DECRO_AudioDs(AudioDataset):
    def postprocess(self):
        self.data["audio_path"] = self.data["filename"].apply(
            lambda x: os.path.join(self.root_path, x)
        )

    def read_metadata_from_txt(self, root_path):
        datas = []
        for txt_file in [
            "ch_dev",
            "ch_eval",
            "ch_train",
            "en_dev",
            "en_eval",
            "en_train",
        ]:
            with open(os.path.join(root_path, txt_file + ".txt"), "r") as f:
                lines = f.readlines()

                for line in lines:
                    line_splits = line.strip().split(" ")
                    filename = f"{txt_file}/{line_splits[1]}.wav"
                    method = line_splits[3]
                    label = 1 if line_splits[-1] == "bonafide" else 0
                    language = txt_file.split("_")[0]
                    split = (
                        txt_file.split("_")[1]
                        .replace("dev", "val")
                        .replace("eval", "test")
                    )
                    datas.append([filename, method, label, language, split])

        data = pd.DataFrame(
            datas, columns=["filename", "method", "label", "language", "split"]
        )
        return data

    def get_vocoder_label(self, x):
        if x["label"] == 1:
            return 0

        method = x["method"]
        for i, t in enumerate(VOCODERS):
            if isinstance(t, list):
                if method in t:
                    return i + 1
            else:
                if method == t:
                    return i + 1
        raise ValueError("Unknown vocoder method", x)

    def _read_metadata(self, root_path, *args, **kwargs):
        """
        read metadatas of audio files from the root_path. We frist extract the metadata
        from the given txt files, then get the fps and length for each audio.

        Args:
            root_path: the root_path for the DECRO.

        """

        data_path = os.path.join(root_path, "dataset_info.csv")
        if os.path.exists(data_path):
            return pd.read_csv(data_path)

        ## Step 1. read all audio paths
        data = self.read_metadata_from_txt(root_path)
        data["audio_path"] = data["filename"].apply(
            lambda x: os.path.join(root_path, x)
        )
        data["vocoder_label"] = data.apply(lambda x: self.get_vocoder_label(x), axis=1)

        data = self.read_audio_info(data)  # read fps and length
        data.to_csv(data_path, index=False)
        return data

    def get_splits(self, language="en"):
        """
            Get train/val/test splits according to the language.

        Args:
            language: 'en' or 'ch'

        Returns:
            Namespace(train, val, test)
        """

        assert language in ["en", "ch"]
        data = self.data.query(f'language == "{language}"')

        sub_datas = []
        for split in ["train", "val", "test"]:
            _data = data.query(f'split == "{split}"').reset_index(drop=True)
            sub_datas.append(_data)

        return Namespace(
            train=sub_datas[0],
            test=sub_datas[-1],
            val=None if len(sub_datas) == 2 else sub_datas[1],
        )

# + tags=["active-ipynb", "style-student"]
# model = DECRO_AudioDs(root_path="/usr/local/ay_data/2-datasets/1-df-audio/DECRO")
# data = model.get_splits(language="en")
#
# data.test
