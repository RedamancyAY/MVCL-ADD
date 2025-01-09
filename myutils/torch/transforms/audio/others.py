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

import numpy as np
import torch


class PadAudioLength:
    """Pad audio to make it have enough number of length

    The input audio must be a 1-D tensor (L,) or 2-D tensor (C, L)
    """

    def __init__(self, length):
        """
        Args:
            length: the wanted audio length
        """
        self.length = length

    def check_input(self, x):
        if len(x.shape) not in [1, 2]:
            raise ValueError(
                "The input audio should be 1-D or 2-D, but with shape of ", x.shape
            )

    def __call__(self, audio, *args, **kwargs):
        """
        Args:
            video: a 1-D tensor (L,) or 2-D tensor (C, L)

        Returns:
            the padded audio.
        """

        self.check_input(audio)

        # set flag to 1_D input
        flag = 0
        if len(audio.shape) == 1:
            audio = audio[None, ...]
            flag = 1

        C, L = audio.shape
        length = self.length

        # check audio length
        if L > length:
            audio = audio[:, :length]
        elif L < length:
            if isinstance(audio, np.ndarray):
                audio = np.concatenate([audio, np.zeros((C, length - L))], axis=1)
            elif isinstance(audio, torch.Tensor):
                audio = torch.concat([audio, torch.zeros(C, length - L)], dim=1)
            else:
                raise ValueError(
                    f"the input video should be a np.ndarray or torch.Tensor, but actually {type(video)}"
                )

        # if input is 1-D, convert the padded audio into 1-D
        if flag:
            audio = audio[0, ...]
        
        return audio

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# x = torch.randn(4880)
# y = np.random.randn(1, 4800)
# module = PadAudioLength(10000)
# module(x).shape, module(y).shape
