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
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange


# -

class VideoToTensor:

    """
    Convert the input 4-D numpy.ndarray video into torch.tensor.
    """

    def __init__(self, input_shape="THWC", output_shape="CTHW", bgr2rgb=False):
        """
        Args:
            input_shape: 'THWC' or 'TCHW'
            output_shape: 'CTHW'
            bgr2rgb: whether to convert the BGR channel into RGB for cv2 reading method
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.bgr2rgb = bgr2rgb



    def check_input(self, x):
        if not isinstance(x, np.ndarray):
            raise ValueError("The input should be a numpy.ndarray, but is a ", type(x))

        if len(x.shape) != 4:
            raise ValueError("The input should be 4-D, but with shape of ", x.shape)

        x_min = np.min(x)
        x_max = np.max(x)
        if x_min < 0 or x_max > 255:
            raise ValueError(
                "The pixel values of the input video should be in [0, 255],"
                f"but actually is [{x_min}, {x_max}]"
            )
    
    def __call__(self, x, *args):
        """
        Args:
            x: a 4-D numpy ndarray
        """

        self.check_input(x)

        x = torch.as_tensor(x)
        x = x / 255
        x = rearrange(
            x, f"{' '.join(self.input_shape)} -> {' '.join(self.output_shape)}"
        )
        if self.bgr2rgb:
            if not "C" in self.output_shape:
                raise ValueError("'C' must in output_shape")
            channel_dim = self.output_shape.index("C")
            if channel_dim == 0:
                x = x[[2, 1, 0], ...]
            else:
                raise NotImplementedError
        return x


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# x = np.random.randint(0, 256, (10, 224, 224, 3))
# y = torch.randn(2, 2)
# module = VideoToTensor(input_shape="THWC", output_shape="CTHW", bgr2rgb=True)
# module(x).shape
# -

class PadVideoFrames:
    """Pad video to make it have enough number of frames

    The input video must be a 4-D tensor with shape (T, H, W, C) or (T, C, H, W)
    """

    def __init__(self, frames):
        """
        Args:
            frames: the wanted number of video frames
            input_shape: 'THWC' or 'TCHW'
        """
        self.frames = frames

    def check_input(self, x):
        if len(x.shape) != 4:
            raise ValueError("The input should be 4-D, but with shape of ", x.shape)


    def __call__(self, video, *args, **kwargs):
        """
        Args:
            video: a 4-D tensor with shape (T, H, W, C) or (T, C, H, W)

        Returns:
            the padded video with shape (frames, H, W, C) or (frames, C, H, W)
        """

        self.check_input(video)
        T, C, H, W = video.shape
        frames = self.frames
        if T > frames:
            video = video[0:frames, ...]
        elif T < frames:
            if isinstance(video, np.ndarray):
                video = np.concatenate([video, np.zeros((frames - T, C, H, W))])
            elif isinstance(video, torch.Tensor):
                video = torch.concat([video, torch.zeros(frames - T, C, H, W)])
            else:
                raise ValueError(f"the input video should be a np.ndarray or torch.Tensor, but actually {type(video)}")
        return video

# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# x = np.random.randint(0, 256, (10, 224, 224, 3))
# y = torch.randn(10, 224, 224 ,3)
# module = PadVideoFrames(frames=512)
# module(x).shape, module(y).shape
