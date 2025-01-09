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
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, roc_curve
from torch import Tensor


# +
def calculate_eer(y, y_score) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Args:
        y: the ture label for prediction
        y_socre: the logits for prediction
    Return:
        thresh, eer, fpr, tpr
    """

    fpr, tpr, thresholds = roc_curve(y, -y_score)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    # return thresh, eer, fpr, tpr
    return 1 - eer


"""
Python compute equal error rate (eer)
ONLY tested on binary classification

:param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
:param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
:param positive_label: the class that is viewed as positive class when computing EER
:return: equal error rate (EER)
"""


def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


# +
import glob
import math
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import io
from scipy.signal import convolve2d
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

# ## SSIM


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode="same"):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, "valid")
    mu2 = filter2(im2, window, "valid")
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, "valid") - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, "valid") - mu2_sq
    sigmal2 = filter2(im1 * im2, window, "valid") - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return np.mean(np.mean(ssim_map))


# ## PSNR


def compute_psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# -

# ## AP & AR

# +
def claculate_ap_at_threshold(
    iou_threshold,
    proposals_list: Union[dict, list],
    fake_periods_list: Union[dict, list],
) -> float:
    values = []
    n_labels = 0

    assert type(proposals_list) == type(fake_periods_list)
    if isinstance(proposals_list, dict):
        T = lambda x: [x[key] for key in sorted(a.keys())]
        proposals_list = T(proposals_list)
        fake_periods_list = T(fake_periods_list)

    for proposals, fake_periods in zip(proposals_list, fake_periods_list):
        proposals = torch.tensor(proposals)
        labels = torch.tensor(fake_periods)
        values.append(get_ap_values(iou_threshold, proposals, labels, 25.0))
        n_labels += len(labels)

    # sort proposals
    values = torch.cat(values)
    ind = values[:, 0].sort(stable=True, descending=True).indices
    values = values[ind]

    # accumulate to calculate precision and recall
    curve = calculate_curve(values, n_labels=n_labels)
    ap = calculate_ap(curve)
    return ap


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors."""

    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.0)
    union_len = len_anchors - inter_len + box_max - box_min
    iou = inter_len / union_len
    return iou


def calculate_curve(values, n_labels):
    acc_TP = 0
    acc_FP = 0
    curve = torch.zeros((len(values), 2))
    for i, (confidence, is_TP) in enumerate(values):
        if is_TP == 1:
            acc_TP += 1
        else:
            acc_FP += 1

        precision = acc_TP / (acc_TP + acc_FP)
        recall = acc_TP / n_labels
        curve[i] = torch.tensor((recall, precision))

    curve = torch.cat([torch.tensor([[1.0, 0.0]]), torch.flip(curve, dims=(0,))])
    return curve


def calculate_ap(curve):
    y_max = 0.0
    ap = 0
    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        x2, y2 = curve[i + 1]
        if y1 > y_max:
            y_max = y1
        dx = x1 - x2
        ap += dx * y_max
    return ap


def get_ap_values(
    iou_threshold: float,
    proposals: Tensor,
    labels: Tensor,
    fps: float,
) -> Tensor:
    n_labels = len(labels)
    ious = torch.zeros((len(proposals), n_labels))
    for i in range(len(labels)):
        ious[:, i] = iou_with_anchors(
            proposals[:, 1] / fps, proposals[:, 2] / fps, labels[i, 0], labels[i, 1]
        )

    # values: (confidence, is_TP) rows
    n_labels = ious.shape[1]
    detected = torch.full((n_labels,), False)
    confidence = proposals[:, 0]
    potential_TP = ious > iou_threshold

    for i in range(len(proposals)):
        for j in range(n_labels):
            if potential_TP[i, j]:
                if detected[j]:
                    potential_TP[i, j] = False
                else:
                    # mark as detected
                    potential_TP[i] = False  # mark others as False
                    potential_TP[i, j] = True  # mark the selected as True
                    detected[j] = True

    is_TP = potential_TP.any(dim=1)
    values = torch.column_stack([confidence, is_TP])
    return values
