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

import torch
import torchvision.io as io

# + tags=["style-activity"]
from .utils import from_float, to_float


# + tags=["active-ipynb"]
# from utils import from_float, to_float
# -

def jpg_compression(img, quality):
    input_dtype = img.dtype
    needs_float = False

    if input_dtype == torch.float32:
        img = from_float(img, dtype=torch.uint8)
        needs_float = True
    elif input_dtype not in (torch.uint8, torch.float32):
        raise ValueError("Unexpected dtype {} for jpg_compression".format(input_dtype))

    img = io.encode_jpeg(img, quality=quality)
    img = io.decode_jpeg(img)

    if needs_float:
        img = to_float(img, max_value=255)
    return img

# # 测试

# + tags=["style-student", "active-ipynb"]
# from io import BytesIO
#
# import numpy as np
# import requests as req
# from myutils.visualization import Plot
# from PIL import Image
#
# img_src = "http://pic.imeitou.com/uploads/allimg/210618/3-21061P92632.jpg"
# response = req.get(img_src)
# image = Image.open(BytesIO(response.content))
# image = np.array(image)
#
# jpg_imgs = []
# for i in range(1, 100, 10):
#     image2 = jpg_compression(torch.tensor(image).permute(2, 0, 1), i)
#     jpg_imgs.append(image2)
#
# Plot.plot_images([image] + jpg_imgs, n_row=3)
# -

# ## 测试时间

# + tags=["active-ipynb"]
# import time
#
# res = 0
# for i in range(10):
#     # x = torch.randint(0, 255, (3, 224, 224)).to(torch.uint8)
#     x = torch.tensor(image).permute(2, 0, 1)
#     s = time.time()
#     y = jpg_compression(x, 30)
#     e = time.time()
#     res += e - s
#     print(e - s, res)
