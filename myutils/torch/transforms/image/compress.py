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
from io import BytesIO
from random import choice, random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile


# +
def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return np.stack([decimg[:, :, 2], decimg[:, :, 1], decimg[:, :, 0]], -1)


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


# -

class Random_CV_PIL_JpegCompression:
    def __init__(self, compress_val, p=0.1):
        self.p = p
        self.compress_val = compress_val

    def compress_3D(self, image, compress_val):
        flag_channel = 0
        if image.shape[-1] > 3:
            image = np.transpose(image, (1, 2, 0))  # (H, W, C)
            flag_channel = 1

        if random() < 0.5:
            res = cv2_jpg(image, compress_val)
        else:
            res = pil_jpg(image, compress_val)

        if flag_channel:
            res = np.transpose(res, (2, 0, 1))
        return res

    def compress_4D(self, image, compress_val):
        """
        image : (T, C, H, W)
        """
        flag_channel = 0
        if image.shape[-1] > 3:
            image = np.transpose(image, (0, 2, 3, 1))
            flag_channel = 1

        if random() < 0.5:
            res = np.stack(
                [cv2_jpg(image[i], compress_val) for i in range(image.shape[0])]
            )
        else:
            res = np.stack(
                [pil_jpg(image[i], compress_val) for i in range(image.shape[0])]
            )

        if flag_channel:
            res = np.transpose(res, (0, 3, 1, 2))
        return res

    def __call__(self, image):
        if random() >= self.p:
            return image
        compress_val = sample_discrete(self.compress_val)

        if type(image) == torch.Tensor:
            image = image.numpy()

        if len(image.shape) == 3:
            return self.compress_3D(image, compress_val)
        else:
            return self.compress_4D(image, compress_val)
        
        return res
# + tags=["active-ipynb"]
# import time
#
# import cv2
#
# img_path = "/usr/local/ay_data/dataset/Set5/head.png"
# img = cv2.imread(img_path)
#
# model = Random_CV_PIL_JpegCompression([30, 40, 50], p=1.0)
# s = time.time()
# res = model(img[0:224, 0:224, :])
# e = time.time()
# print(e - s)
# print(torch.from_numpy(res).shape)
#
# # test 4D tensor
# s = time.time()
# x = img[0:224, 0:224, :]
# model(np.stack([x]*10*8))
# e = time.time()
# print(e - s)
