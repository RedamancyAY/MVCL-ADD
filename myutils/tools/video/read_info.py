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

import argparse
import cv2


# # Metadata

# ## FPS、 length、height & width

# + editable=true slideshow={"slide_type": ""}
def read_video_fps_len_height_width(path):
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return FPS, frame_count, frame_height, frame_width


# + editable=true slideshow={"slide_type": ""}
def read_video_height_width(path):
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_height, frame_width


# + editable=true slideshow={"slide_type": ""}
def read_video_fps_len(path):
    cap = cv2.VideoCapture(path)
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return FPS, frame_count

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# from myutils.tools import check_dir, read_file_paths_from_folder
# from tqdm import tqdm
#
# wav_paths = read_file_paths_from_folder("/home/ay/data/DATA/dataset/2-audiovisual/VGG-Sound", exts=["mp4"])
#
# paths = wav_paths[111100:111300]
#
# for path in tqdm(paths):
#     _ = read_video_fps_len_height_width(path)
