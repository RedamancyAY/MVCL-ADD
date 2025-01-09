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
import argparse
import os
import random
import sys

import numpy as np
import torch


# -

def make_model(cfg_file, cfg, args):
    """build models from cfg file name and model cfg

    Args:
        cfg_file: the file name of the model cfg, such as "LCNN/wavefake"
        cfg: the model config

    """
    if cfg_file.startswith("MultiView/"):
        from .MultiView import MultiViewModel_lit
        model = MultiViewModel_lit(cfg=cfg.MODEL, args=args)
    return model

