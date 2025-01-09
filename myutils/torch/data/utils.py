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

import os
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.nn import functional as F


def get_gt_iou_map(
    frames: int,
    video_labels: List,
    temporal_scale: int,
    fps: int = 25,
    max_duration: int = 25,
    padding: int = 512,
):
    """generate gt_iou_map from fake periods


    Take video as example.

    Args:
        frames: 
            total video frames
        video_labels: 
            fake periods, for example, [[0.5, 1.2], [2.5, 3.2]]
        temporal_scale: 
            total video frames
        max_duration: 
            maximum number of video frames in a fake period
        padding: 
            padding the gt_iou_map if total frames less than padding

    Returns:
        a groundtruth iou map tensor `gt_iou_map`.

        `gt_iou_map` is with shape of [max_duration, padding], where gt_iou_map[i, j]
        denotes the iou score between the fake periods and a video clip whose frames
        ranging from [j, j+i].


    """
    corrected_second = frames / fps
    temporal_gap = 1 / temporal_scale

    ################################################################################
    # change the measurement from second to percentage
    gt_bbox = []
    for j in range(len(video_labels)):
        tmp_start = max(min(1, video_labels[j][0] / corrected_second), 0)
        tmp_end = max(min(1, video_labels[j][1] / corrected_second), 0)
        gt_bbox.append([tmp_start, tmp_end])

    ###############################################################################
    # generate R_s and R_e
    gt_bbox = torch.tensor(gt_bbox)
    if len(gt_bbox) > 0:
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
    else:
        gt_xmins = np.array([])
        gt_xmaxs = np.array([])
    ##############################################################################

    gt_iou_map = torch.zeros([max_duration, temporal_scale])
    if len(gt_bbox) > 0:
        for begin in range(temporal_scale):
            for duration in range(max_duration):
                end = begin + duration
                if end > temporal_scale:
                    break
                gt_iou_map[duration, begin] = torch.max(
                    iou_with_anchors(
                        begin * temporal_gap,
                        (end + 1) * temporal_gap,
                        gt_xmins,
                        gt_xmaxs,
                    )
                )
                # [i, j]: Start in i, end in j.

    #############################################################################
    gt_iou_map = F.pad(gt_iou_map.float(), pad=[0, padding - frames, 0, 0])
    return gt_iou_map


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    
    copy from: https://github.com/ControlNet/LAV-DF/blob/60c75dcc587ae38a67d14d2122c2d9739cd3a016/utils.py#L116
    
    """

    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.0)
    union_len = len_anchors - inter_len + box_max - box_min
    iou = inter_len / union_len
    return iou

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# x = get_gt_iou_map(
#     frames = 108,
#     video_labels=[[0.15, 1.1], [3.5, 4.0]],
#     temporal_scale= 108
# )
# print(x)

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# x = get_gt_iou_map(
#     frames = 108,
#     video_labels=[],
#     temporal_scale= 108
# )
# print(x.shape)
