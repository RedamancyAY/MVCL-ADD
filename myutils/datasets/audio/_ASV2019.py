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

VOCODERs = ["-"] + ["A{:02d}".format(i) for i in range(1, 20)]


class ASV2019LA_AudioDs(AudioDataset):

    def update_audio_path(self, data, root_path):
        data["audio_path"] = data["relative_path"].apply(
            lambda x: os.path.join(root_path, x)
        )
    
    def postprocess(self):
        self.update_audio_path(self.data, self.root_path)
        self.vocoders = VOCODERs

    def read_label_data(self, root_path):
        label_files = [
            "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
            "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        ]
        
        label_data = pd.concat(
            [
                pd.read_csv(
                    os.path.join(root_path, _f),
                    delimiter=" ",
                    names=["speaker", "filename", "-", "method", "label"],
                )
                for _f in label_files
            ],
            ignore_index=True,
        )
        ### convert label 'bonafide' -> 1, 'spoof' -> 0
        label_data['label'] = label_data['label'].replace('bonafide', 1).replace('spoof', 0)

        return label_data
    
    def _read_metadata(self, root_path, *args, **kwargs):
        paths = read_file_paths_from_folder(root_path, exts="flac")
        data = pd.DataFrame(paths, columns=["path"])
        data["relative_path"] = data["path"].apply(lambda x: x.replace(root_path + '/', ''))
        data["filename"] = data["path"].apply(lambda x: os.path.split(x)[1].replace(".flac", ""))
        data['split'] = data["path"].apply(lambda x: 'train' if '_train' in x else ('val' if '_dev' in x else 'test'))
        

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

# root_path = "/home/ay/data/0-原始数据集/ASV2019"
# ds = ASV2019LA_AudioDs(root_path=root_path)
# data = ds.data
# splits = ds.get_splits()

# +
# data.query("label == 1").groupby(['compression', 'split']).count()

# +
# from IPython.display import HTML, display

# display(HTML(splits.test[0].groupby(["vocoder", "attack", "subset"]).count().to_html()))
# -
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>path</th>
#       <th>relative_path</th>
#       <th>filename</th>
#       <th>split</th>
#       <th>speaker</th>
#       <th>-</th>
#       <th>method</th>
#       <th>vocoder_label</th>
#       <th>audio_path</th>
#       <th>audio_fps</th>
#       <th>audio_len</th>
#     </tr>
#     <tr>
#       <th>label</th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>108978</td>
#       <td>108978</td>
#       <td>108978</td>
#       <td>108978</td>
#       <td>108978</td>
#       <td>108978</td>
#       <td>108978</td>
#       <td>108978</td>
#       <td>108978</td>
#       <td>108978</td>
#       <td>108978</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>12483</td>
#       <td>12483</td>
#       <td>12483</td>
#       <td>12483</td>
#       <td>12483</td>
#       <td>12483</td>
#       <td>12483</td>
#       <td>12483</td>
#       <td>12483</td>
#       <td>12483</td>
#       <td>12483</td>
#     </tr>
#   </tbody>
# </table>
#

# ds.data

# +
# print(ds.data.groupby('label').count().to_html())
# -


# <center><img src="https://raw.githubusercontent.com/RedamancyAY/CloudImage/main/img202408111630054.png" width="500"/></center>
#
#
# 在训练集和测试集中，只包含A01-A06方法，测试集中是A07-A19方法

# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th></th>
#       <th>path</th>
#       <th>relative_path</th>
#       <th>filename</th>
#       <th>speaker</th>
#       <th>-</th>
#       <th>method</th>
#       <th>vocoder_label</th>
#       <th>audio_path</th>
#       <th>audio_fps</th>
#       <th>audio_len</th>
#     </tr>
#     <tr>
#       <th>split</th>
#       <th>label</th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th rowspan="2" valign="top">test</th>
#       <th>0</th>
#       <td>63882</td>
#       <td>63882</td>
#       <td>63882</td>
#       <td>63882</td>
#       <td>63882</td>
#       <td>63882</td>
#       <td>63882</td>
#       <td>63882</td>
#       <td>63882</td>
#       <td>63882</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>7355</td>
#       <td>7355</td>
#       <td>7355</td>
#       <td>7355</td>
#       <td>7355</td>
#       <td>7355</td>
#       <td>7355</td>
#       <td>7355</td>
#       <td>7355</td>
#       <td>7355</td>
#     </tr>
#     <tr>
#       <th rowspan="2" valign="top">train</th>
#       <th>0</th>
#       <td>22800</td>
#       <td>22800</td>
#       <td>22800</td>
#       <td>22800</td>
#       <td>22800</td>
#       <td>22800</td>
#       <td>22800</td>
#       <td>22800</td>
#       <td>22800</td>
#       <td>22800</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>2580</td>
#       <td>2580</td>
#       <td>2580</td>
#       <td>2580</td>
#       <td>2580</td>
#       <td>2580</td>
#       <td>2580</td>
#       <td>2580</td>
#       <td>2580</td>
#       <td>2580</td>
#     </tr>
#     <tr>
#       <th rowspan="2" valign="top">val</th>
#       <th>0</th>
#       <td>22296</td>
#       <td>22296</td>
#       <td>22296</td>
#       <td>22296</td>
#       <td>22296</td>
#       <td>22296</td>
#       <td>22296</td>
#       <td>22296</td>
#       <td>22296</td>
#       <td>22296</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>2548</td>
#       <td>2548</td>
#       <td>2548</td>
#       <td>2548</td>
#       <td>2548</td>
#       <td>2548</td>
#       <td>2548</td>
#       <td>2548</td>
#       <td>2548</td>
#       <td>2548</td>
#     </tr>
#   </tbody>
# </table>

# +
# print(ds.data.groupby(['split', 'label']).count().to_html())
