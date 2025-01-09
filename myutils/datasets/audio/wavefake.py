# %load_ext autoreload
# %autoreload 2

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

from myutils.tools import read_file_paths_from_folder, to_list
from myutils.datasets.base import AudioDataset

# ## Dataset Preparation

# 1. Uncompress the wavefake dataset, rename it into 'WaveFake'
# 2. change the folder `WaveFake/common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech` into 'WaveFake/ljspeech_tts'
#     * Inside `WaveFake/ljspeech_tts`, there are directly 16283 audios, but the folder `WaveFake/ljspeech_tts/generated` still have 16283 audios. We delete the `generated` folder.
# 3. Uncompress the LJSeech dataset, rename it into `ljspeech_real` and put it in the `WaveFake` folder.
# 4. Uncompress the JSUT dataset, rename it into `jsut_real` and put it in the `WaveFake` folder.
#
# The folder sturcture of WaveFake is: 
# ```json
# WaveFake
# ├── jsut_multi_band_melgan
# ├── jsut_parallel_wavegan
# ├── jsut_real
# ├── ljspeech_full_band_melgan
# ├── ljspeech_hifiGAN
# ├── ljspeech_melgan
# ├── ljspeech_melgan_large
# ├── ljspeech_multi_band_melgan
# ├── ljspeech_parallel_wavegan
# ├── ljspeech_real
# ├── ljspeech_tts
# ├── ljspeech_waveglow
# └── readme.txt
#
# 12 directories, 1 file
# ```

#
# 每个文件夹下的音频数量如下：
# | trainSet   | method            |   path |
# |:-----------|:------------------|-------:|
# | jsut       | multi_band_melgan |   5000 |
# | jsut       | parallel_wavegan  |   5000 |
# | jsut       | real              |   5000 |
# | ljspeech   | full_band_melgan  |  13100 |
# | ljspeech   | hifiGAN           |  13100 |
# | ljspeech   | melgan            |  13100 |
# | ljspeech   | melgan_large      |  13100 |
# | ljspeech   | multi_band_melgan |  13100 |
# | ljspeech   | parallel_wavegan  |  13100 |
# | ljspeech   | real              |  13100 |
# | ljspeech   | tts               |  16283 |
# | ljspeech   | waveglow          |  13100 |

# ## WaveFake class

# `WaveFake` will read the metadata info for all the audios, and save it (csv format) in the root_path of WaveFake. The examples of the metadata are showed as:
#
# |        | path                                                                                         | trainSet   | method            |   fps |   length |   label | id               |
# |-------:|:---------------------------------------------------------------------------------------------|:-----------|:------------------|------:|---------:|--------:|:-----------------|
# |  15321 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_tts/gen_13607.wav                    | ljspeech   | tts               | 22050 |  3.20435 |       0 | gen_13607        |
# |  91937 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_melgan/LJ024-0083_gen.wav            | ljspeech   | melgan            | 22050 |  3.25079 |       0 | LJ024-0083       |
# |  43707 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_waveglow/LJ018-0097.wav              | ljspeech   | waveglow          | 22050 |  5.7005  |       0 | LJ018-0097       |
# |  75366 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_parallel_wavegan/LJ043-0150_gen.wav  | ljspeech   | parallel_wavegan  | 22050 |  4.69043 |       0 | LJ043-0150       |
# | 121075 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_melgan_large/LJ003-0283_gen.wav      | ljspeech   | melgan_large      | 22050 |  8.85841 |       0 | LJ003-0283       |
# |   4522 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_tts/gen_7735.wav                     | ljspeech   | tts               | 22050 |  6.33905 |       0 | gen_7735         |
# | 106158 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/jsut_multi_band_melgan/BASIC5000_4225_gen.wav | jsut       | multi_band_melgan | 24000 |  8.9875  |       0 | BASIC5000_4225   |
# |   1361 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_tts/gen_14993.wav                    | ljspeech   | tts               | 22050 |  4.51628 |       0 | gen_14993        |
# | 106225 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/jsut_multi_band_melgan/BASIC5000_4962_gen.wav | jsut       | multi_band_melgan | 24000 |  2.85    |       0 | BASIC5000_4962   |
# | 125639 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_multi_band_melgan/LJ033-0121_gen.wav | ljspeech   | multi_band_melgan | 22050 |  4.55111 |       0 | LJ033-0121       |
# |  51593 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_waveglow/LJ031-0180.wav              | ljspeech   | waveglow          | 22050 |  7.53488 |       0 | LJ031-0180       |
# |  90079 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_melgan/LJ021-0099_gen.wav            | ljspeech   | melgan            | 22050 |  8.71909 |       0 | LJ021-0099       |
# |  87734 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_melgan/LJ002-0280_gen.wav            | ljspeech   | melgan            | 22050 |  9.8917  |       0 | LJ002-0280       |
# |  71944 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_hifiGAN/LJ043-0122_generated.wav     | ljspeech   | hifiGAN           | 22050 | 10.0078  |       0 | LJ043-0122erated |
# |  16546 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_real/wavs/LJ031-0084.wav             | ljspeech   | real              | 22050 |  7.87868 |       1 | LJ031-0084       |m

VOCODERs = [
    "melgan",
    "parallel_wavegan",
    "multi_band_melgan",
    "full_band_melgan",
    "hifiGAN",
    "melgan_large",
    "waveglow",
    "tts"
]
TRAINSETs = ["ljspeech", "jsut"]


class WaveFake_AudioDs(AudioDataset):
    def postprocess(self):
        self.data["audio_path"] = self.data["file"].apply(
            lambda x: os.path.join(self.root_path, x)
        )
        self.vocoders = VOCODERs

        
    
    def get_corpus_vocoder_from_path(self, path):
        """
        Get the names of corpus and vocoders from the audio path.
        For example: Given the path "/home/ay//data/DATA/2-datasets/1-df-audio/WaveFake/jsut_real/tmp.wav",
        it will extract the corpus `jsut` and the method `real`.

        Args:
            paths: the audio path

        Returns:
            the corpus and the vocoder method
        """
        p = r"WaveFake/([a-zA-Z]+)\_(\w+)/"
        match = re.search(p, path)
        corpus = match.group(1)
        vocoder = match.group(2)
        return [corpus, vocoder]

    def _read_metadata(self, root_path, *args, **kwargs):
        
        wav_paths = read_file_paths_from_folder(root_path, exts=["wav"])
        
        print(root_path, len(wav_paths))
        
        
        data = pd.DataFrame()
        data["audio_path"] = wav_paths
        data["file"] = [path.replace(root_path + "/", "") for path in wav_paths]
        data["label"] = data["audio_path"].apply(
            lambda x: 1 if "ljspeech_real" in x or "jsut_real" in x else 0
        )

        data[["corpus", "method"]] = data.apply(
            lambda x: tuple(self.get_corpus_vocoder_from_path(x["audio_path"])),
            axis=1,
            result_type="expand",
        )

        data["language"] = data["corpus"].apply(
            lambda x: "English" if x == "ljspeech" else "Japanese"
        )
        data["vocoder_label"] = data["method"].apply(
            lambda x: 0 if x == "real" else (8 if x == "tts" else VOCODERs.index(x) + 1)
        )

        data["id"] = data["audio_path"].apply(
            lambda x: os.path.basename(x)
            .replace(".wav", "")
            .replace("_generated", "")
            .replace("_gen", "")
        )

        data = self.read_audio_info(data)  # read fps and length
        return data

    def _get_sub_data(self, corpus, method):
        """
        Given the corpus of Vocoders and the vocoder method, return the subdata
        Args:
            trainSet: the dataset for training the Vocoders
            method: the vocoder method
        """
        # print(trainSet, method)
        if isinstance(corpus, int):
            corpus = TRAINSETs[corpus]
        if isinstance(method, int):
            method = VOCODERs[method]

        data = self.data
        sub_data = data[(data["corpus"] == corpus) & (data["method"] == method)].copy()

        if method == 'tts':
            sub_data["vocoder_label"] = 0
        
        return sub_data.reset_index(drop=True)

    def get_sub_data(
        self, corpus: [list, str], methods: [list, str], contain_real=True
    ) -> pd.DataFrame:
        corpus = to_list(corpus)
        methods = to_list(methods)
        if contain_real:
            methods = methods + ["real"]
        
        data = []
        for _corpus in corpus:
            for _method in methods:
                _data = self._get_sub_data(_corpus, _method)
                data.append(_data)
        data = pd.concat(data).reset_index(drop=True)
        
        return data

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# dataset = WaveFake_AudioDs(
#     root_path="/usr/local/ay_data/2-datasets/1-df-audio/WaveFake"
# )
# data = dataset.get_sub_data(corpus=0, methods=[0, 1, 2, 3, 4, 5, 6])
# splits = [64_000, 16_000, 24_800]
# datas = dataset.split_data(data, splits)
