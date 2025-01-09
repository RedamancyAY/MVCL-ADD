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
from random import choice, random

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


# +
def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def _blur(image, sigma):
    sigma = sample_continuous(sigma)
    return _blur_with_simga(image, sigma)


def _blur_with_simga(image, sigma):
    ndim = len(image.shape)
    assert ndim == 3
    for i in range(image.shape[-1]):
        gaussian_filter(image[:, :, i], output=image[:, :, i], sigma=sigma)
    return image


# -

class Gaussian_blur:
    def __init__(self, sigma, p=0.1):
        self.sigma = sigma
        self.p = p

    @classmethod
    def gaussian_blur(cls, image, sigma):
        _blur(image, sigma)

    def blur_3D(self, image, sigma=-1):
        flag_channel = 0
        if image.shape[-1] > 3:
            image = np.transpose(image, (1, 2, 0))  # (H, W, C)
            flag_channel = 1

        if sigma == -1:
            res = _blur(image, self.sigma)
        else:
            res = _blur_with_simga(image, sigma)

        if flag_channel:
            res = np.transpose(res, (2, 0, 1))
        return res

    def blur_4D(self, image):
        """
        image : (T, C, H, W)
        """
        flag_channel = 0
        if image.shape[-1] > 3:
            image = np.transpose(image, (0, 2, 3, 1))
            flag_channel = 1

        sigma = sample_continuous(self.sigma)
        res = np.stack(self.blur_3D(image[i], sigma) for i in range(image.shape[0]))

        if flag_channel:
            res = np.transpose(res, (0, 3, 1, 2))
        return res

    def __call__(self, image):
        if random() >= self.p:
            return image

        if type(image) == torch.Tensor:
            image = image.numpy()

        if len(image.shape) == 4:
            return self.blur_4D(image)
        else:
            return self.blur_3D(image)

        flag_channel = 0
        if image.shape[-1] > 3:
            image = np.transpose(image, (1, 2, 0))
            flag_channel = 1

        res = _blur(image, self.sigma)

        if flag_channel:
            res = np.transpose(res, (2, 0, 1))
        return res

# + tags=["active-ipynb"]
# import time
#
# import cv2
#
# img_path = "/usr/local/ay_data/dataset/Set5/baby.png"
# img = cv2.imread(img_path)
#
# # test class method
# s = time.time()
# Gaussian_blur.gaussian_blur(img[0:224, 0:224, :], [1, 3])
# e = time.time()
# print(e - s)
#
# # test __call__ method
# blurer = Gaussian_blur(sigma=[1, 3], p=1)
# s = time.time()
# blurer(img[0:224, 0:224, :])
# e = time.time()
# print(e - s)
#
# # test 4D tensor
# blurer = Gaussian_blur(sigma=[1, 3], p=1)
# s = time.time()
# x = img[0:224, 0:224, :]
# blurer(np.stack([x]*10*8))
# e = time.time()
# print(e - s)
