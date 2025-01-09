# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from typing import List, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
# -

from .functional import claculate_ap_at_threshold


# ## AP & AR

class AP(Module):
    """
    Average Precision

    The mean precision in Precision-Recall curve.
    """

    def __init__(
        self, iou_thresholds: Union[float, List[float]] = 0.5, tqdm_pos: int = 1
    ):
        super().__init__()
        self.iou_thresholds: List[float] = (
            iou_thresholds if isinstance(iou_thresholds, list) else [iou_thresholds]
        )
        self.ap: dict = {}

    def forward(
        self, fake_periods_list: Union[dict, list], proposals_list: Union[dict, list]
    ) -> dict:
        for iou_threshold in self.iou_thresholds:
            self.ap[f'AP@{iou_threshold}'] = claculate_ap_at_threshold(
                iou_threshold,
                proposals_list=proposals_list,
                fake_periods_list=fake_periods_list,
            )

        return self.ap


