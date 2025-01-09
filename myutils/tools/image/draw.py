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

import cv2
import numpy as np


# +

def draw_rectangle_on_img(img, sh, sw, eh, ew):
    """

    draw a rectangle on the input image.

    Args:
        img: an image with size (H, W, C)
        sh: H维度上的起点
        sw: W维度上的起点
        eh: H维度上的终点
        ew: W维度上的终点
    """
    if len(img.shape) == 2:
        img = img[:,:,None]
    
    h, w, c = img.shape
    assert len(img.shape) == 3 and c in [1, 3]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def set_rgb(x1, y1, x2, y2, r, g, b):
        img_rgb[x1:y1, x2:y2, 0] = r
        img_rgb[x1:y1, x2:y2, 1] = g
        img_rgb[x1:y1, x2:y2, 2] = b

    rr, gg, bb = 255, 0, 0
    set_rgb(sh, eh, sw - 4, sw, rr, gg, bb)
    set_rgb(sh, eh, ew, ew + 4, rr, gg, bb)
    set_rgb(sh - 4, sh, sw - 4, ew + 4, rr, gg, bb)
    set_rgb(eh, eh + 4, sw - 4, ew + 4, rr, gg, bb)
    img_crop = img_rgb[sh:eh, sw:ew, :]
    return img_rgb, img_crop

