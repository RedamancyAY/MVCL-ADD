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

VOCODERs = ["bonafide"] + ["A{:02d}".format(i) for i in range(7, 20)]


class ASV2021LA_AudioDs(AudioDataset):

    def update_audio_path(self, data, root_path):
        data["audio_path"] = data["relative_path"].apply(
            lambda x: os.path.join(root_path, x)
        )
    
    def postprocess(self):
        self.update_audio_path(self.data, self.root_path)
        self.vocoders = VOCODERs

    def read_label_data(self, root_path):
        label_files = [
            "keys/LA/CM/trial_metadata.txt"
        ]
        
        label_data = pd.concat(
            [
                pd.read_csv(
                    os.path.join(root_path, _f),
                    delimiter=" ",
                    names=["speaker", "filename", "codec", 'col1', "method", "label", 'trim', 'split'],
                )
                for _f in label_files
            ],
            ignore_index=True,
        )
        ### convert label 'bonafide' -> 1, 'spoof' -> 0
        label_data['label'] = label_data['label'].replace('bonafide', 1).replace('spoof', 0)
        label_data['split'] = label_data['split'].replace('progress', 'train').replace('eval', 'test')

        ### randomly select 2000 samples from training split for validation.
        val_data = label_data.query("split == 'train'").sample(2000, random_state=42)
        label_data.loc[val_data.index, "split"] = "val"
        
        return label_data
    
    def _read_metadata(self, root_path, *args, **kwargs):
        paths = read_file_paths_from_folder(root_path, exts="flac")
        data = pd.DataFrame(paths, columns=["path"])
        data["relative_path"] = data["path"].apply(lambda x: x.replace(root_path + '/', ''))
        data["filename"] = data["path"].apply(lambda x: os.path.split(x)[1].replace(".flac", ""))
        
        label_data = self.read_label_data(root_path)
        data = pd.merge(data, label_data)

        data["vocoder_label"] = data["method"].apply(lambda x: VOCODERs.index(x))

        self.update_audio_path(data, root_path)
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

        return Namespace(
            train=sub_datas[0],
            val=sub_datas[1],
            test=sub_datas[2],
        )

# +
# root_path = "/home/ay/data/0-原始数据集/ASV2021-LA"
# ds = ASV2021LA_AudioDs(root_path=root_path)
# data = ds.data
# splits = ds.get_splits()
# +
# data.groupby(['split', 'label']).count()

# data.groupby(['split', 'method']).count()
# -




