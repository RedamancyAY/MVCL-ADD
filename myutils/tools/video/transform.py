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

# %load_ext autoreload
# %autoreload 2

import argparse
import os
import cv2
import numpy as np
import torchvision
from contextlib import contextmanager

from .read import yield_video_frames, read_video_clip_cv2

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# from read import yield_video_frames, read_video_clip_cv2

# +
from contextlib import contextmanager

@contextmanager
def video_writer(filename, codec='XVID', fps=20.0, frame_size=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

    try:
        yield out
    finally:
        out.release()


# -

def resize_video_file(video_path, dest_path, frame_size, fps=25):

    if os.path.exists(dest_path):
        return 2
    
    if isinstance(frame_size, int):
        frame_size = (frame_size, frame_size)
    H, W = frame_size
    
    with video_writer(dest_path, codec='mp4v', fps=fps, frame_size=(W, H)) as out:
        for frame in yield_video_frames(video_path, bgr2rgb=False):
            new_frame = cv2.resize(frame, (W, H))
            out.write(new_frame)
    return 1


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# video_path='/home/ay/data/0-原始数据集/LAV-DF/test/000018.mp4'
# dest_path ='test.mp4'
#
# resize_video_file(video_path, dest_path, frame_size=(96, 96))

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# v1 = read_video_clip_cv2(video_path)
# v2 = read_video_clip_cv2('test.mp4')
#
# v1.shape, v2.shape
# -

from myutils.tools import read_file_paths_from_folder

video_paths = read_file_paths_from_folder('/home/ay/data/0-原始数据集/LAV-DF', exts='mp4')

import pandas as pd

data = pd.DataFrame(video_paths, columns=['path'])


def resize_video(video_path):
    dest_path = video_path.replace(".mp4", '-96x96.mp4')
    resize_video_file(video_path, dest_path, frame_size=(96, 96))
    return 1


# +
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=10)

# df.apply(func)
data.parallel_apply(lambda x:resize_video(x['path']), axis=1)
