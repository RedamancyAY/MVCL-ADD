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

"""Image related operations
"""

# +
import math
import os
import re
import sys
from io import BytesIO
from typing import Sequence

import cv2
import numpy as np
import requests
from PIL import Image
# -

from turbojpeg import TurboJPEG
JPEG_READER = None


# # 读取图像

# !pip install  pyTurboJPEG

def read_web_image(
    src: str = "https://z3.ax1x.com/2021/08/27/hQFHjx.jpg",
) -> np.ndarray:
    """读取网络图像

    Args:
        src: The web url of the image

    Returns:
        a numpy ndarray
    """
    response = requests.get(src)
    image = Image.open(BytesIO(response.content))
    image = np.array(image)
    return image


def read_grey_image(img_path: str) -> np.ndarray:
    """read grey-scale image

    If the image is a color image, it will be conveted into grey scale.

    Args:
        img_path(str): the path of image

    Returns:
        a numpy ndarray
    
    """
    img_org = Image.open(img_path)
    img = img_org.convert('L')
    img = np.array(img)
    return img


def read_rgb_image(img_path):
    global JPEG_READER
    if JPEG_READER is None:
        JPEG_READER = TurboJPEG()
    with open(img_path, 'rb') as file:
        image = JPEG_READER.decode(file.read(), 0)  # decode raw image
    return image


# +
# def read_rgb_image(img_path: str) -> np.ndarray:
#     """read rgb-scale image

#     Args:
#         img_path(str): the path of image

#     Returns:
#         a numpy ndarray 
#     """
#     img_org = Image.open(img_path)
#     img = img_org.convert('RGB')
#     img = np.array(img)
#     return img
# -

def read_image(img_path: str) -> np.ndarray:
    """read image using PIL

    Args:
        img_path(str): the path of image

    Returns:
        a numpy ndarray 
    """
    img_org = Image.open(img_path)
    img = np.array(img_org)
    return img

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# img_path = '/home/ay/data/DATA/dataset/2-audiovisual/VGG-Sound/imgs/00_do2b96M8_000110/000001.jpg'
#
# img = read_image(img_path)
# rgb_img = read_rgb_image(img_path)
# grey_img = read_grey_image(img_path)
# print(rgb_img.shape, grey_img.shape, img.shape)
