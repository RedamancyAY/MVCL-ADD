# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=[] editable=true slideshow={"slide_type": ""}
# %load_ext autoreload
# %autoreload 2

# + tags=[] editable=true slideshow={"slide_type": ""}
import functools
import logging
import math
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


# + tags=[]
class SampleTransformDataset(torch.utils.data.Dataset):
    """
    Apple transform for each sample of the dataset. It will apple transform at the end of
    the `__get_item__()` method.
    """

    def __init__(self, transforms=None, *kwargs):
        """
        Args:
            transforms: a dict, {'key1':transform1, 'key2':transform2}
        """
        self.transforms = transforms

    def apply_transforms(self, res):
        # print(self.transforms)
        if self.transforms is None:
            return res

        try:
            for key in self.transforms:
                # print(key)
                res[key] = self.transforms[key](res[key])
        except KeyError:
            raise KeyError("Your transforms do not have the key", key)
        return res
