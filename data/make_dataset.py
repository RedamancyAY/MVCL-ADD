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
from argparse import Namespace
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import Compose

# -

from myutilsdatasets.audio import (
    ASV2019LA_AudioDs,
    ASV2021_AudioDs,
    ASV2021LA_AudioDs,
    DECRO_AudioDs,
    InTheWild_AudioDs,
    LibriSeVoc_AudioDs,
    MLAAD_AudioDs,
    VGGSound_AudioDs,
    WaveFake_AudioDs,
)
from myutilstools import color_print
from myutilstorch.transforms.audio import AudioRawBoost, SpecAugmentTransform_Wave
from myutilstorchaudio.transforms import LFCC, RandomNoise, RawBoost, RandomBackgroundNoise

# + editable=true slideshow={"slide_type": ""}
try:
    # from .datasets import ADD2023, LAV_DF_Audio, LibriSeVoc, WaveFake, DECRO
    from .tools import WaveDataset
except ImportError:
    # from datasets import ADD2023, LAV_DF_Audio, LibriSeVoc, WaveFake, DECRO
    from tools import WaveDataset


# -

# # Make audio splits (DataFrame)


def get_emotion_labels(
    data: pd.DataFrame,
    emotion_df_path="/home/ay/data/DATA/2-datasets/1-df-audio/emotions.csv",
):
    emotions = pd.read_csv(emotion_df_path)
    emotions["emotion_label"] = emotions["index"]
    emotions = emotions[["audio_path", "emotion_label"]]
    data = pd.merge(data, emotions, how="left", on="audio_path")
    return data


# ## Different datasets


def get_InTheWild_data(
    root_path="/home/ay/data/DATA/2-datasets/1-df-audio/release_in_the_wild",
):
    dataset = InTheWild_AudioDs(root_path=root_path)
    return dataset.data


# +
# dataset = InTheWild_AudioDs(root_path="/home/ay/data/DATA/2-datasets/1-df-audio/release_in_the_wild")
# dataset.data.groupby('label').count()
# -

# ### WaveFake


def get_WaveFake_JP(root_path="/home/ay/data/DATA/2-datasets/1-df-audio/WaveFake"):
    dataset = WaveFake_AudioDs(root_path)
    data_train = dataset.get_sub_data(corpus=1, methods=[1,2])
    train, val = dataset.split_data(data_train, splits=[0.8, 0.2], return_list=True, refer="id")
    data_splits = Namespace(train=train, val=val)
    return data_splits


def make_WaveFake(cfg):
    dataset = WaveFake_AudioDs(root_path=cfg.root_path)
    # dataset.data = get_emotion_labels(dataset.data)

    if cfg.task == "inner_eval":
        color_print("WaveFake task: inner evaluation")
        data = dataset.get_sub_data(corpus=cfg.corpus, methods=cfg.methods)
        data_splits = dataset.split_data(data, splits=cfg.splits, refer="id")
    elif cfg.task == "cross_lang":
        color_print("WaveFake task: cross language evaluation")

        task_cfg = cfg.task_cfg
        data_train = dataset.get_sub_data(corpus=task_cfg.train.corpus, methods=task_cfg.train.methods)
        train, val = dataset.split_data(data_train, splits=task_cfg.train.splits, return_list=True, refer="id")
        test = dataset.get_sub_data(corpus=task_cfg.test.corpus, methods=task_cfg.test.methods)
        data_splits = Namespace(train=train, val=val, test=test)
    elif cfg.task == "cross_method":
        color_print("WaveFake task: cross method")

        task_cfg = cfg.task_cfg
        # get real data, and split it into train/val/test
        data_real = dataset._get_sub_data(task_cfg.train.corpus, "real")
        real_train, real_val, real_test = dataset.split_data(
            data_real, splits=[0.6, 0.2, 0.2], return_list=True, refer="id"
        )

        data_train = dataset.get_sub_data(
            corpus=task_cfg.train.corpus,
            methods=task_cfg.train.methods,
            contain_real=False,
        )
        train, val = dataset.split_data(data_train, splits=task_cfg.train.splits, return_list=True, refer="id")
        test = [
            dataset.get_sub_data(corpus=_cfg.corpus, methods=_cfg.methods, contain_real=False) for _cfg in task_cfg.test
        ]
        train = pd.concat([train, real_train], ignore_index=True)
        val = pd.concat([val, real_val], ignore_index=True)
        test = [pd.concat([_test, real_test], ignore_index=True) for _test in test]
        data_splits = Namespace(train=train, val=val, test=test)

    return data_splits


# ### LibriSeVoc


def make_LibriSeVoc(cfg):
    dataset = LibriSeVoc_AudioDs(root_path=cfg.ROOT_PATHs.LibriSeVoc)
    # dataset.data = get_emotion_labels(dataset.data)

    if cfg.task == "inner_eval":
        color_print("LibriSeVoc task: inner evaluation")

        data = dataset.get_sub_data(methods=cfg.methods)
        data_splits = dataset.split_data(data, splits=cfg.splits, refer="id")
    elif cfg.task == "cross_method":
        color_print("LibriSeVoc task: cross method evaluation")
        task_cfg = cfg.task_cfg

        # get real data, and split it into train/val/test
        data_real = dataset.get_sub_data([], contain_real=True)
        real_train, real_val, real_test = dataset.split_data(
            data_real, splits=[0.6, 0.2, 0.2], return_list=True, refer="id"
        )

        data_train = dataset.get_sub_data(methods=task_cfg.train.methods, contain_real=False)
        train, val = dataset.split_data(data_train, splits=task_cfg.train.splits, return_list=True, refer="id")
        test = [dataset.get_sub_data(methods=_cfg.methods, contain_real=False) for _cfg in task_cfg.test]
        train = pd.concat([train, real_train], ignore_index=True)
        val = pd.concat([val, real_val], ignore_index=True)
        test = [pd.concat([_test, real_test], ignore_index=True) for _test in test]

        data_splits = Namespace(train=train, val=val, test=test)
    elif cfg.task == "cross_dataset":
        color_print("LibriSeVoc task: cross dataset evaluation")
        task_cfg = cfg.task_cfg
        data_train = dataset.get_sub_data(methods=task_cfg.train.methods)
        train, val = dataset.split_data(data_train, splits=task_cfg.train.splits, return_list=True, refer="id")
        test = []
        for _cfg in task_cfg.test:
            if _cfg.dataset.lower() == "wavefake":
                dataset2 = WaveFake_AudioDs(root_path=cfg.ROOT_PATHs.WaveFake)
                _data = dataset2.get_sub_data(corpus=_cfg.corpus, methods=_cfg.methods)
                test.append(_data)
        test.append(get_InTheWild_data())
        test.append(get_DECRO_test_splits(language="en"))
        test.append(get_DECRO_test_splits(language="cn"))
        # test += get_ASV2021_test_splits()
        data_splits = Namespace(train=train, val=val, test=test)
    return data_splits


# ### DECRO


# +
def get_DECRO_test_splits(root_path="/home/ay/data/DATA/2-datasets/1-df-audio/DECRO", language="en"):
    dataset = DECRO_AudioDs(root_path=root_path)
    en_splits = dataset.get_splits(language="en")
    ch_splits = dataset.get_splits(language="ch")
    if language == "en":
        data = en_splits.test
    else:
        data = ch_splits.test

    data["vocoder_label_org"] = data["vocoder_label"]
    data["vocoder_label"] = 0
    return data
    
def get_DECRO_splits(root_path="/home/ay/data/DATA/2-datasets/1-df-audio/DECRO"):
    dataset = DECRO_AudioDs(root_path=root_path)
    en_splits = dataset.get_splits(language="en")
    ch_splits = dataset.get_splits(language="ch")
    return en_splits, ch_splits


# -

def make_DECRO(cfg):
    dataset = DECRO_AudioDs(root_path=cfg.root_path)
    # dataset.data = get_emotion_labels(dataset.data)

    en_splits = dataset.get_splits(language="en")
    ch_splits = dataset.get_splits(language="ch")

    if cfg.task == "en->ch":
        color_print("DECRO task: en->ch")
        train, val, test = (
            en_splits.train,
            en_splits.val,
            [en_splits.train, ch_splits.test, get_InTheWild_data()],
        )
    else:
        color_print("DECRO task: ch->en")
        train, val, test = (
            ch_splits.train,
            ch_splits.val,
            [ch_splits.test, en_splits.test, get_InTheWild_data()],
        )
    data_splits = Namespace(train=train, val=val, test=test)
    return data_splits


# ### ASV 2019


# +
def make_ASV2019(cfg):
    dataset = ASV2019LA_AudioDs(root_path=cfg.root_path)
    if cfg.task == "inner_eval":
        color_print("ASVspoof 2021 task: inner evaluation")
        data_splits = dataset.get_splits()

    data_splits.test = [data_splits.test]
    data_splits.test += get_ASV2021_whole_test_split()
    data_splits.test += get_ASV2021_test_splits()
    return data_splits


def get_ASV2019_test_split(root_path="/home/ay/data/0-原始数据集/ASV2019"):
    dataset = ASV2019LA_AudioDs(root_path=root_path)
    data_splits = dataset.get_splits()
    return data_splits.test


# -

# ### ASV 2021


# +
def get_ASV2021_test_splits(root_path="/home/ay/ASV2021"):
    dataset = ASV2021_AudioDs(root_path=root_path)
    data_splits = dataset.get_test_splits()
    return data_splits


def get_ASV2021_whole_test_split(root_path="/home/ay/ASV2021"):
    dataset = ASV2021_AudioDs(root_path=root_path)
    test = dataset.get_whole_test_split()
    return [test]


# -


def make_ASV2021(cfg):
    dataset = ASV2021_AudioDs(root_path=cfg.root_path)
    if cfg.task == "inner_eval":
        color_print("ASVspoof 2021 DF task: inner evaluation")
        data_splits = dataset.get_splits()

    data_splits.test += get_ASV2021_whole_test_split()
    return data_splits


def make_ASV2021_LA(cfg):
    dataset = ASV2021LA_AudioDs(root_path=cfg.root_path)
    if cfg.task == "inner_eval":
        color_print("ASVspoof 2021 LA task: inner evaluation")
        data_splits = dataset.get_splits()

    data_splits.test = [data_splits.test]
    data_splits.test.append(get_ASV2019_test_split())
    data_splits.test += get_ASV2021_whole_test_split()
    data_splits.test += get_ASV2021_test_splits()
    return data_splits


# ### VGG Sound


def make_VGGSound(cfg):
    dataset = VGGSound_AudioDs(root_path=cfg.root_path)
    color_print("VGGSound: load splits")
    data_splits = dataset.get_splits()  # only train and test splits
    data_splits.val = data_splits.test
    return data_splits


# ### MLAAD


def make_MLAAD(cfg):


    def append_test(_data_splits):
        _data_splits.test.append(get_InTheWild_data())
        _data_splits.test.append(get_DECRO_test_splits(language="en"))
        _data_splits.test.append(get_DECRO_test_splits(language="cn"))
    

    
    dataset = MLAAD_AudioDs(root_path=cfg.root_path)
    color_print("MLAAD: load splits")
    if cfg.task == "cross_lang":
        data_splits = dataset.get_splits(language_list=['en', 'de', 'es'])
        append_test(data_splits)
    elif cfg.task == "de_es_ru":
        print("MLDDA, load DE, ES, RU subsets for training!!!!!!!")
        data_splits = dataset.get_splits(language_list=['de', 'es', 'ru'])
        append_test(data_splits)
    elif cfg.task == "DECRO":
        print("MLDDA, load DECRO and WaveFake subsets for training!!!!!!!")
        data_splits = dataset.get_splits(language_list=[])
        append_test(data_splits)
        en_splits, ch_splits = get_DECRO_splits()
        jp_splits = get_WaveFake_JP()
        data_splits.train = pd.concat([en_splits.train, ch_splits.train, jp_splits.train], ignore_index=True)
        data_splits.val = pd.concat([en_splits.val, ch_splits.val, jp_splits.val], ignore_index=True)
    return data_splits

# ## Dict

MAKE_DATASETS = {
    "WaveFake": make_WaveFake,
    "LibriSeVoc": make_LibriSeVoc,
    "DECRO": make_DECRO,
    "ASV2021": make_ASV2021,
    "ASV2021_LA": make_ASV2021_LA,
    "ASV2019_LA": make_ASV2019,
    "VGGSound": make_VGGSound,
    "MLAAD": make_MLAAD,
}


# # Build DataLoaders


def build_feature(cfg):
    if cfg.audio_feature == "LFCC":
        return LFCC()
    return None


# ## Transform

from myutilstorchaudio.transforms import RandomAudioCompression
from myutilstorchaudio.transforms.self_operation import (
    AudioToTensor,
    CentralAudioClip,
    RandomAudioClip,
    RandomPitchShift,
    RandomSpeed,
)

from audio_augmentations import *


def build_transforms(cfg=None, args=None):
    # t1 = RandomNoise(snr_min_db=10.0, snr_max_db=120.0, p=1.0)
    # # t = RawBoost(algo=[5], p=0.5)
    # t2 = RandomSpeed(min_speed=0.5, max_speed=2.0, p=1.0)
    # t3 = RandomPitchShift(p=1.0)

    # sr = 16000
    # num_samples=48000
    # transforms = [
    #     RandomResizedCrop(n_samples=num_samples),
    #     RandomApply([PolarityInversion()], p=0.8),
    #     RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
    #     RandomApply([Gain()], p=0.2),
    #     HighLowPass(sample_rate=sr), # this augmentation will always be applied in this aumgentation chain!
    #     RandomApply([Delay(sample_rate=sr)], p=0.5),
    #     RandomApply([PitchShift(
    #         n_samples=num_samples,
    #         sample_rate=sr
    #     )], p=0.4),
    #     RandomApply([Reverb(sample_rate=sr)], p=0.3)
    # ]

    # return {
    #     "train": transforms,
    #     "val": [
    #         CentralAudioClip(length=48000),
    #         AudioToTensor(),
    #     ],
    # }

    res = {
        "train": [
            # RandomSpeed(min_speed=0.5, max_speed=2.0, p=0.5),
            # RandomAudioCompression(p=0.9),
            # RandomSpeed(min_speed=0.5, max_speed=2.0, p=1.0),
            RandomAudioClip(length=48000),
            RandomNoise(snr_min_db=10.0, snr_max_db=120.0, p=1.0),
            AudioToTensor(),
            RandomApply([PitchShift(n_samples=48000, sample_rate=16000)], p=0.5),
            # RandomPitchShift(p=0.5),
        ],
        "val": [
            CentralAudioClip(length=48000),
            AudioToTensor(),
        ],
    }

    
    
    if args is not None and args.test_noise:
        res["test_noise"] = [
            CentralAudioClip(length=48000),
            RandomBackgroundNoise(
                16000,
                noise_dir="/home/ay/data/0-原始数据集/musan/noise",
                p=1.0,
                min_snr_db=args.test_noise_level,
                max_snr_db=args.test_noise_level,
                noise_type=args.test_noise_type,
            ),
            AudioToTensor(),
        ]

    # if args.cfg.startswith('MPE_LCNN'):
    #     from myutilstorchaudio.transforms import MPE_LFCC
    #     for key in res:
    #         res[key].append(MPE_LFCC())
    
    return res


# ## Common Opeations


def build_dataloader(data: pd.DataFrame, cfg, is_training: bool = True, args=None):
    transforms = build_transforms(cfg.transforms, args=args)
    transform = transforms["train"] if is_training else transforms["val"]

    _ds = WaveDataset(
        data,
        sample_rate=cfg.sample_rate,
        normalize=True,
        transform=transform,
        dtype="tensor",
    )

    if not is_training and cfg.test_batch_size > 0:
        batch_size = cfg.test_batch_size
    else:
        batch_size = cfg.batch_size

    _dl = DataLoader(
        _ds,
        batch_size=batch_size,
        # num_workers=cfg.num_workers,
        num_workers=20,
        pin_memory=True,
        shuffle=True if is_training else False,
        # shuffle=True,
        prefetch_factor=2,
        drop_last=True if is_training else False,
    )
    return _ds, _dl


# ## Door


def over_sample_dataset(data, column="label"):
    n_fake = len(data[data[column] == 0])
    n_real = len(data[data[column] == 1])
    if n_fake == n_real:
        return data
    if n_fake > n_real:
        sampled = data[data[column] == 1].sample(n=n_fake - n_real, replace=True)
        balanced_data = pd.concat([data, sampled])
    else:
        sampled = data[data[column] == 0].sample(n=n_real - n_fake, replace=True)
        balanced_data = pd.concat([data, sampled])

    balanced_data = balanced_data.copy().reset_index(drop=True)
    return balanced_data


def print_audio_splits_label_distribution(audio_splits):
    res = {}
    for _split in ["train", "val", "test"]:
        _data = getattr(audio_splits, _split)
        res[_split] = ""
        if isinstance(_data, list):
            for _data2 in _data:
                tmp = _data2.groupby("label").count()
                num_0 = tmp.loc[0][0] if 0 in tmp.index else 0
                num_1 = tmp.loc[1][0] if 1 in tmp.index else 0
                res[_split] += f" {num_0}/{num_1}"
        else:
            tmp = _data.groupby("label").count()
            num_0 = tmp.loc[0][0] if 0 in tmp.index else 0
            num_1 = tmp.loc[1][0] if 1 in tmp.index else 0
            res[_split] += f" {num_0}/{num_1}"

    color_print(f"Fake/Real label distribution in train/val/test: {res['train']}, {res['val']}, {res['test']}")


def make_data(cfg, args=None):
    # make audio splits (pd.DataFrame)
    audio_splits = MAKE_DATASETS[cfg.name](cfg.dataset_cfg)
    audio_splits.train = over_sample_dataset(audio_splits.train, column="label")

    print_audio_splits_label_distribution(audio_splits)

    # make dataset and dataloaders
    train_ds, train_dl = build_dataloader(audio_splits.train, cfg, is_training=True, args=args)
    train_ds2, train_dl2 = build_dataloader(audio_splits.train, cfg, is_training=False, args=args)
    val_ds, val_dl = build_dataloader(audio_splits.val, cfg, is_training=False, args=args)
    if isinstance(audio_splits.test, list):
        test_ds, test_dl = [], []
        for _test in audio_splits.test:
            _ds, _dl = build_dataloader(_test, cfg, is_training=False, args=args)
            test_ds.append(_ds)
            test_dl.append(_dl)
    else:
        test_ds, test_dl = build_dataloader(audio_splits.test, cfg, is_training=False, args=args)

    # collect all dataloaders
    ds = Namespace(train=train_ds, val=val_ds, test=test_ds, train_wo_transform=train_ds2)
    dl = Namespace(train=train_dl, val=val_dl, test=test_dl, train_wo_transform=train_dl2)

    print(args)
    if args is not None and args.test_noise:
        color_print("!!!!Test robustness: Load audio with background noise")
        test_noise = build_transforms(args=args)["test_noise"]
        if isinstance(dl.test, list):
            for _dl in dl.test:
                _dl.dataset.transform = test_noise
            # dl.test = dl.test[1]
        else:
            dl.test.transform = test_noise

    return ds, dl
