# +
import os
import random
import re
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

# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>audio_path</th>
#       <th>vocoder</th>
#       <th>path</th>
#       <th>language</th>
#       <th>vocoder_label</th>
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
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>76000</td>
#       <td>76000</td>
#       <td>76000</td>
#       <td>76000</td>
#       <td>76000</td>
#       <td>76000</td>
#       <td>76000</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>64875</td>
#       <td>64875</td>
#       <td>64875</td>
#       <td>64875</td>
#       <td>64875</td>
#       <td>64875</td>
#       <td>64875</td>
#     </tr>
#   </tbody>
# </table>

VOCODERs = [
    "none",
    "capacitron-t2-c50",
    "facebook/mms-tts-deu",
    "facebook/mms-tts-eng",
    "facebook/mms-tts-fin",
    "facebook/mms-tts-fra",
    "facebook/mms-tts-hun",
    "facebook/mms-tts-nld",
    "facebook/mms-tts-ron",
    "facebook/mms-tts-rus",
    "facebook/mms-tts-swe",
    "facebook/mms-tts-ukr",
    "fast_pitch",
    "glow-tts",
    "griffin_lim",
    "jenny",
    "microsoft/speecht5_tts",
    "neural_hmm",
    "overflow",
    "parler_tts",
    "suno/bark",
    "tacotron-DDC",
    "tacotron2",
    "tacotron2-DCA",
    "tacotron2-DDC",
    "tacotron2-DDC_ph",
    "tortoise-v2",
    "vits",
    "vits--neon",
    "vits-neon",
    "xtts_v1.1",
    "xtts_v2",
]

# +


def read_metadata_of_MLADD(root_path):
    ## Step 1: two folders
    fake_folder = os.path.join(root_path, "fake")
    real_folder = os.path.join(root_path, "real")

    ## Step 2: read all csv from MLAAD
    files = read_file_paths_from_folder(fake_folder, exts=["csv"])
    datas = []
    for csv in files:
        datas.append(pd.read_csv(csv, delimiter="|"))
    data = pd.concat(datas, ignore_index=True)

    ## Step 3: from MLAAD metdata, get all the used real audios
    real_data = data.copy()
    real_data = real_data.drop_duplicates("original_file").reset_index(drop=True)
    real_data["label"] = 1
    real_data["path"] = real_data["original_file"]
    real_data["audio_path"] = real_data["path"].apply(lambda x: os.path.join(real_folder, x))
    real_data["vocoder"] = "none"
    real_data["language"] = real_data["path"].apply(
        lambda x: x.split("/")[0][:2]
    )  # en_UK, en_US => en, de_DE => de, ...

    ## Step 4: from MLAAD metdata, deal all the fake metadata
    fake_data = data.copy()
    fake_data["label"] = 0
    fake_data["audio_path"] = fake_data["path"].apply(lambda x: os.path.join(fake_folder, x[7:]))
    fake_data["vocoder"] = fake_data["architecture"]
    fake_data = fake_data.reset_index(drop=True)

    ## Step 5: combine all the real and fake metadata
    columns = ["audio_path", "label", "vocoder", "path", "language"]
    data = pd.concat([real_data[columns], fake_data[columns]], ignore_index=True)
    data["vocoder_label"] = data["vocoder"].apply(lambda x: VOCODERs.index(x))
    return data


# +
# root_path = "/mnt/data/zky/0-原始数据集/MLADD"
# data = read_metadata_of_MLADD(root_path)
# print(data.groupby(["language", "label"]).count().to_html())
# -

# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th></th>
#       <th>audio_path</th>
#       <th>vocoder</th>
#       <th>path</th>
#       <th>vocoder_label</th>
#     </tr>
#     <tr>
#       <th>language</th>
#       <th>label</th>
#       <th></th>
#       <th></th>
#       <th></th>
#       <th></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>ar</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th>bg</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th>cs</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th>da</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th rowspan="2" valign="top">de</th>
#       <th>0</th>
#       <td>6000</td>
#       <td>6000</td>
#       <td>6000</td>
#       <td>6000</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>5856</td>
#       <td>5856</td>
#       <td>5856</td>
#       <td>5856</td>
#     </tr>
#     <tr>
#       <th>el</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th rowspan="2" valign="top">en</th>
#       <th>0</th>
#       <td>19000</td>
#       <td>19000</td>
#       <td>19000</td>
#       <td>19000</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>31239</td>
#       <td>31239</td>
#       <td>31239</td>
#       <td>31239</td>
#     </tr>
#     <tr>
#       <th rowspan="2" valign="top">es</th>
#       <th>0</th>
#       <td>4000</td>
#       <td>4000</td>
#       <td>4000</td>
#       <td>4000</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>3913</td>
#       <td>3913</td>
#       <td>3913</td>
#       <td>3913</td>
#     </tr>
#     <tr>
#       <th>et</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th>fi</th>
#       <th>0</th>
#       <td>2000</td>
#       <td>2000</td>
#       <td>2000</td>
#       <td>2000</td>
#     </tr>
#     <tr>
#       <th rowspan="2" valign="top">fr</th>
#       <th>0</th>
#       <td>6000</td>
#       <td>6000</td>
#       <td>6000</td>
#       <td>6000</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>5821</td>
#       <td>5821</td>
#       <td>5821</td>
#       <td>5821</td>
#     </tr>
#     <tr>
#       <th>ga</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th>hi</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th>hu</th>
#       <th>0</th>
#       <td>3000</td>
#       <td>3000</td>
#       <td>3000</td>
#       <td>3000</td>
#     </tr>
#     <tr>
#       <th rowspan="2" valign="top">it</th>
#       <th>0</th>
#       <td>7000</td>
#       <td>7000</td>
#       <td>7000</td>
#       <td>7000</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>6708</td>
#       <td>6708</td>
#       <td>6708</td>
#       <td>6708</td>
#     </tr>
#     <tr>
#       <th>mt</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th>nl</th>
#       <th>0</th>
#       <td>4000</td>
#       <td>4000</td>
#       <td>4000</td>
#       <td>4000</td>
#     </tr>
#     <tr>
#       <th rowspan="2" valign="top">pl</th>
#       <th>0</th>
#       <td>4000</td>
#       <td>4000</td>
#       <td>4000</td>
#       <td>4000</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>3808</td>
#       <td>3808</td>
#       <td>3808</td>
#       <td>3808</td>
#     </tr>
#     <tr>
#       <th>ro</th>
#       <th>0</th>
#       <td>2000</td>
#       <td>2000</td>
#       <td>2000</td>
#       <td>2000</td>
#     </tr>
#     <tr>
#       <th rowspan="2" valign="top">ru</th>
#       <th>0</th>
#       <td>4000</td>
#       <td>4000</td>
#       <td>4000</td>
#       <td>4000</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>3710</td>
#       <td>3710</td>
#       <td>3710</td>
#       <td>3710</td>
#     </tr>
#     <tr>
#       <th>sk</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th>sw</th>
#       <th>0</th>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#       <td>1000</td>
#     </tr>
#     <tr>
#       <th rowspan="2" valign="top">uk</th>
#       <th>0</th>
#       <td>4000</td>
#       <td>4000</td>
#       <td>4000</td>
#       <td>4000</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>3820</td>
#       <td>3820</td>
#       <td>3820</td>
#       <td>3820</td>
#     </tr>
#   </tbody>
# </table>

class MLAAD_AudioDs(AudioDataset):

    default_root_path = "/mnt/data/zky/0-原始数据集/MLADD"
    
    def postprocess(self):
        fake_folder = os.path.join(self.root_path, "fake")
        real_folder = os.path.join(self.root_path, "real")

        self.data["audio_path"] = self.data["path"].apply(
            lambda x: os.path.join(fake_folder, x[7:]) if x.startswith("./fake") else os.path.join(real_folder, x)
        )

    def _read_metadata(self, root_path, *args, **kwargs):
        """
        read metadatas of audio files from the root_path. We frist extract the metadata
        from the given txt files, then get the fps and length for each audio.

        Args:
            root_path: the root_path for the DECRO.

        """
        data = read_metadata_of_MLADD(root_path)
        data = self.read_audio_info(data)  # read fps and length
        
        return data

    def get_splits(self, language_list=['en', 'de', 'es']):
        """
            Get train/val/test splits according to the language.

        Args:
            language: 'en' or 'ch'

        Returns:
            Namespace(train, val, test)
        """

        in_domain_data = self.data.query(f"language in {language_list}").reset_index(drop=True)
        if len(in_domain_data) > 10:
            train, val, in_domain_test = self.split_data(in_domain_data, splits=[0.8, 0.1, 0.1], return_list=True)
        else:
            train, val, in_domain_test = None, None, None
        
        out_domain_data = self.data.query(f"language not in {language_list}").reset_index(drop=True)
        grouped = out_domain_data.groupby('language')
        dfs = {category: group for category, group in grouped}
        test_data = [dfs[key].reset_index(drop=True) for key in dfs.keys()]
        test_data.insert(0, out_domain_data)
        
        return Namespace(
            train=train,
            test=test_data,
            test_keys = ['full'] + list(dfs.keys()),
            val=val,
        )

# +
# root_path = "/mnt/data/zky/0-原始数据集/MLADD"
# ds = MLAAD_AudioDs(root_path)
# # ds.get_splits(language_list=[])

# +
# data = ds.data
# used = ['en', 'de', 'es', 'fr', 'it', 'pl', 'ru', 'uk']
# data['is_used'] = data['language'].apply(lambda x: 1 if x in used else 0)
# -

# ![|600](https://raw.githubusercontent.com/RedamancyAY/CloudImage/main/img202408132227950.png)

# ![|600](https://raw.githubusercontent.com/RedamancyAY/CloudImage/main/img202408132228811.png)

# +
# # Group by Language and count unique vocoders
# vocoder_count = data.groupby('language')['vocoder'].nunique().reset_index(name='Num_Vocoders')

# # Display the result
# print(vocoder_count)

# +
# # Group by Language and count unique vocoders
# vocoder_count = data.groupby('is_used')['vocoder'].nunique().reset_index(name='Num_Vocoders')

# # Display the result
# print(vocoder_count)
