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

"""从视频中提取视频帧
"""

# %load_ext autoreload
# %autoreload 2

# +
import os
import random
from typing import Union

import ffmpeg
# -

from .read_info import read_video_fps_len


def extract_frames_from_video(
    input_path: str, output_dir: str, fps: Union[int, None] = None
) -> bool:
    """提取视频帧到一个文件夹里

    视频帧的格式为`%06d.jpg`

    Args:
        input_path: the video path
        output_dir: the output dir, the frames will be 'output_dir/%06d.jpg'
        fps: the extracted frames per second. if `None`, it will be set to
            be the video fps; besides, it cannot be larger than video_fps.

    Returns:
        bool: True if successful, False otherwise.

    Raises:
        FileNotFoundError: the output_dir cannot be create
    """


    ## Add a output option of ffmpeg to control the output fps.
    ## If fps is None, output all frames
    
    options = {}
    if fps is not None:
        video_fps, video_len = read_video_fps_len(input_path)
        if video_fps < fps:  # the fps cannot be larger than video_fps
            print(
                f"video fps is {video_fps}, but the input extracting fps is {fps}"
                f". So We Set the extracting fps to video fps: {video_fps} "
            )
            fps = video_fps
        options = {'r':fps}


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    try:
        out, _ = (
            ffmpeg.input(input_path)
            .output(f"{output_dir}/%06d.jpg", **options)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except ffmpeg.Error as e:
        print(f"stderr when converting {input_path} :", e.stderr.decode("utf8"))
        return False
    return True

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# input_path = "/Users/ay/Downloads/videos/test.mp4"
# output_dir = os.path.splitext(input_path)[0]
#
# extract_frames_from_video(input_path, output_dir, fps=1)
