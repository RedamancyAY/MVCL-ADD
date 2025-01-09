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

from yacs.config import CfgNode as ConfigurationNode

from argparse import Namespace
from typing import Any, NamedTuple


# # 默认配置

def NameCfgNode(**kwargs):
    x = ConfigurationNode(kwargs)
    return x


ALL_DATASETS = ["WaveFake"]

ROOT_PATHs = NameCfgNode(
    WaveFake="/home/ay/data/DATA/2-datasets/1-df-audio/WaveFake",
    LibriSeVoc="/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc",
    DECRO="/home/ay/data/DATA/2-datasets/1-df-audio/DECRO",
    Wild = "/home/ay/data/DATA/2-datasets/1-df-audio/release_in_the_wild",
    # ASV2021 = "/home/ay/data/ASVspoof2021_DF_eval",
    ASV2021 = "/home/ay/ASV2021",
    ASV2021_LA = "/home/ay/data/0-原始数据集/ASV2021-LA",
    ASV2019_LA = "/home/ay/data/0-原始数据集/ASV2019",
    VGGSound = "/home/ay/data/DATA/2-datasets/4-audio/VGGSound",
    MLAAD = "/home/ay/data/0-原始数据集/MLADD",
    Codecfake = "/home/ay/data2/Codecfake16k"
)

# ### WaveFake

WaveTasks = {
    "inner_eval": NameCfgNode(
        corpus=0, methods=[0, 1, 2, 3, 4, 5, 6], splits=[64_000, 16_000, 24_800]
    ),
    "cross_lang": NameCfgNode(
        train=NameCfgNode(corpus=0, methods=[1, 2], splits=[0.8, 0.2]),
        test=NameCfgNode(corpus=1, methods=[1, 2], splits=[1.0]),
    ),
    "cross_method": NameCfgNode(
        train=NameCfgNode(corpus=0, methods=[0, 1], splits=[0.8, 0.2]),
        test=[
            NameCfgNode(corpus=0, methods=[i], splits=[1.0]) for i in [2, 3, 4, 5, 6, 7]
        ],
    )
}


def WaveFake(task="inner_eval"):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.root_path = ROOT_PATHs.WaveFake
    C.task = task
    if task == "inner_eval":
        task = WaveTasks[task]
        C.corpus = task.corpus  # 0 / 1
        C.methods = task.methods  # 0-6
        C.splits = task.splits
    else:
        try:
            C.task_cfg = WaveTasks[task]
        except KeyError:
            raise KeyError("Error task name for WaveFake")
    return C


# ### LibriSeVoc

LibriSeVocTasks = {
    "inner_eval": NameCfgNode(
        methods=[0, 1, 2, 3, 4, 5], splits=[55_440, 18_480, 18_487]
    ),
    "cross_method": NameCfgNode(
        train=NameCfgNode(methods=[0, 4], splits=[0.8, 0.2]),
        test=[NameCfgNode(methods=[i], splits=[1.0]) for i in [1, 2, 3, 5]],
    ),
    "cross_dataset": NameCfgNode(
        train=NameCfgNode(methods=[0, 1, 2, 3, 4, 5], splits=[0.8, 0.2]),
        test=[
            NameCfgNode(dataset="WaveFake", corpus=0, methods=[i], splits=[1.0])
            for i in [0, 1, 2, 3, 4, 5, 6]
        ],
    ),
}


def LibriSeVoc(task="inner_eval"):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.task = task
    if task == "inner_eval":
        task = LibriSeVocTasks[task]
        C.methods = task.methods  # 0-5
        C.splits = task.splits
    else:
        try:
            C.task_cfg = LibriSeVocTasks[task]
        except KeyError:
            raise KeyError("Error task name for LibriSeVoc")
    return C


# ### DECRO 

DECRO_Tasks = {
    "inner_eval": NameCfgNode(
        corpus=0, methods=[0, 1, 2, 3, 4, 5, 6], splits=[64_000, 16_000, 24_800]
    ),
    "cross_lang": NameCfgNode(
        train=NameCfgNode(corpus=0, methods=[1, 2], splits=[0.8, 0.2]),
        test=NameCfgNode(corpus=1, methods=[1, 2], splits=[1.0]),
    ),
    "cross_method": NameCfgNode(
        train=NameCfgNode(corpus=0, methods=[0, 5], splits=[0.8, 0.2]),
        test=[
            NameCfgNode(corpus=0, methods=[i], splits=[1.0]) for i in [1, 2, 3, 4, 6]
        ],
    )
}


def DECRO(task=None):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.root_path = ROOT_PATHs.DECRO
    C.task = task
    C.main = "en" if task == "en->ch" else "ch"
    return C


# ### ASV2019

def ASV2019_LA(task=None):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.root_path = ROOT_PATHs.ASV2019_LA
    C.task = task
    return C


# ### ASV2021

# +
def ASV2021(task=None):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.root_path = ROOT_PATHs.ASV2021
    C.task = task
    return C
    
def ASV2021_LA(task=None):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.root_path = ROOT_PATHs.ASV2021_LA
    C.task = task
    return C


# -

def VGGSound(task=None):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.root_path = ROOT_PATHs.VGGSound
    C.task = task
    return C


# ### MLAAD

def MLAAD(task=None):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.root_path = ROOT_PATHs.MLAAD
    C.task = task
    return C


# ## Codecfake

def Codecfake(task=None):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.root_path = ROOT_PATHs.Codecfake
    C.task = task
    return C


# # Settings

DATASETs = {
    'WaveFake' : WaveFake,
    'LibriSeVoc': LibriSeVoc,
    'DECRO': DECRO,
    'ASV2019_LA' : ASV2019_LA,
    'ASV2021' : ASV2021,
    'ASV2021_LA' : ASV2021_LA,
    'VGGSound' : VGGSound,
    'MLAAD' : MLAAD,
    'Codecfake' : Codecfake,
}


def get_dataset_cfg(name, task, __C=None):
    if __C is None:
        __C = ConfigurationNode()

    __C.dataset_cfg = DATASETs[name](task)

    __C.sample_rate = 16000  # audio sampling ratio
    __C.max_wave_length = 48000  # audio length for training
    __C.batch_size = 16  # batch size
    __C.test_batch_size = -1  # batch size
    __C.num_workers = 10  # number of worker to load dataloaders

    __C.transforms = None
    
    return __C


