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
import os, sys
import cv2
import numpy as np
import torchvision
from torchvision.io import read_video

from myutils.tools.image import read_rgb_image

from .errors import CV2CannotOpenVideoError


def yield_video_frames(video_path, start_frame=0, end_frame=None, bgr2rgb=True):
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        raise CV2CannotOpenVideoError("Error: Could not open video: " + video_path)
        return None
    
    frame_num = start_frame

    while True:
        # 逐帧读取视频
        ret, frame = cap.read()
        # 如果读取成功，ret为True，否则为False
        if not ret:
            break
        if bgr2rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
        
        frame_num += 1

        if end_frame is not None and frame_num == end_frame:
            break


def read_video_clip_cv2(video_path, start_frame=0, end_frame=None):
    """use cv2 to read video clips

    Args:
        video_path: the path of the input video
        start_frame: starting frame to read
        end_frame: ending frame to read

    Returns:
        the read video clip (rgb!).
    """
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        raise CV2CannotOpenVideoError("Error: Could not open video: " + video_path)
        return None

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_num = start_frame

    frames = []
    while True:
        # 逐帧读取视频
        ret, frame = cap.read()
        # 如果读取成功，ret为True，否则为False
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_num += 1

        if end_frame is not None and frame_num == end_frame:
            break

    video = np.stack(frames)
    
    return video


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# video = read_video_clip_cv2(video_path='/home/ay/data/0-原始数据集/LAV-DF/test/000018.mp4')
# from myutils.visualization import Plot
# Plot.plot_images([video[0]])
# -

def read_video_cv2(video_path):
    """use cv2 to read the whole video

    implemented by above read_video_clip_cv2.

    Args:
        video_path: the path of the input video

    Returns:
        the read rgb video.
    """
    
    return read_video_clip_cv2(video_path, start_frame=0, end_frame=None)


def read_video_clip_from_frames(video_img_path, start_frame=0, end_frame=None):
    """read a video clip from the total frames of a video

    The video frames are extracted into a folder, where the frame image format is
    `%06d.jpg`. For example, '000000.jpg', '000001.jpg'

    Args:
        video_img_path: the folder that contain all the video frames
        start_frame: the start frame to read
        end_frame: the end frame to read

    Returns:
        a numpy array with shape of (T, H, W, C)
    
    """

    img_paths = os.listdir(video_img_path)
    img_paths = [x for x in img_paths if x.endswith('.jpg')]

    if end_frame is None:
        end_frame = len(img_paths) - 1

    imgs = []
    for i in range(start_frame, end_frame + 1):
        img_path = os.path.join(video_img_path, '%06d.jpg'%(i+1))
        img = read_rgb_image(img_path)
        imgs.append(img)

    if len(imgs) is None:
        return None
    return np.stack(imgs, axis=0)



def read_video_from_frames(video_img_path):
    """read a video from its extracted frames

    The video frames are extracted into a folder, where the frame image format is
    `%06d.jpg`. For example, '000000.jpg', '000001.jpg' ...

    
    Args:
        video_img_path: the folder that contain all the video frames

    Returns:
        a numpy array with shape of (T, H, W, C)
    """
    return read_video_clip_from_frames(video_img_path, start_frame=0, end_frame=None)


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# img_path = '/home/ay/data/DATA/dataset/2-audiovisual/VGG-Sound/imgs/00_do2b96M8_000110'
# read_video_clip_from_frames(img_path, start_frame=0, end_frame=4).shape
# -

def read_stepped_random_video_clip_cv2(
    video_path, start_frame=0, step_size=25, clip_frames=5
):
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_num = start_frame

    frames = []
    while True:
        # 逐帧读取视频
        ret, frame = cap.read()
        # 如果读取成功，ret为True，否则为False
        if not ret:
            break

        if (frame_num - start_frame) % step_size == 0:
            frames.append(frame)

        frame_num += 1

        if len(frames) == clip_frames:
            break

    video = np.stack(frames)
    # if video.shape[-1] == 3:
    # video = video[:,:,:,::-1]
    return video

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# video_path = "/home/ay/data/DATA/dataset/2-audiovisual/VGG-Sound/video/-0legLrA6Ns.mp4"
# video, _, _ = read_video(video_path, start_pts=0, end_pts=None, pts_unit="sec")
# print(video.shape)
#
# read_video_cv2(video_path).shape
#
# read_random_video_clip_cv2(video_path, start_frame=10, end_frame=10 + 25).shape
# print(
#     read_stepped_random_video_clip_cv2(
#         video_path, start_frame=10, clip_frames=10, step_size=25
#     ).shape
# )
