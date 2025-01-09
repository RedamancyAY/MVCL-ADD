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

# + tags=["active-ipynb"] editable=true slideshow={"slide_type": ""}
# %load_ext autoreload
# %autoreload 2

# + tags=[]
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from torchvision.transforms import Normalize

# + tags=[]
try:
    import clip
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install clip by: pip install git+https://github.com/openai/CLIP.git"
    )


# + tags=[]
class CLIP_ImageEncoder(nn.Module):
    """The image encoder of the CLIP

    CLIP: https://github.com/openai/CLIP.git
    CLIP_ImageEncoder delete the text encoder and projection layer.

    Attributes:
        normalize: whether to normalize the input image
        del_text_encoder: default True. delete the text encoder and projection layer.


    """

    def __init__(self, normalize=False, backbone='RN50', del_text_encoder=True):
        super().__init__()

        # The normalization is from the original code of CLIP
        self.normalize = normalize
        if normalize:
            self.img_normalize = Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            )
            
        # load pretrained CLIP from specified backbone
        self.CLIP, _ = clip.load(backbone, device="cpu")
        if del_text_encoder:
            del self.CLIP.transformer
            del self.CLIP.text_projection

    def check_input(self, x):
        """check the input

        The input image must be a 3-channel color image.
        
        Args:
            x: (B, C, H, W)
            
        Returns:
            the original input
        """
        C = x.shape[1]
        if C != 3:
            raise ValueError(
                "The channel number of the input tensor should be 3, but ", x.shape
            )
        return x

    def preprocess(self, x):
        """preprocess the input image
        
        First, check the input shape; Then, judge whether to normalize the image.
        
        Args:
            x: (B, C, H, W)
            
        Returns:
            a tensor with same shape of input
        """
        x = self.check_input(x)
        if self.normalize:
            x = self.img_normalize(x)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        x = self.CLIP.encode_image(x)
        return x
    
    def encode_video(self, x):
        B = x.shape[0]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.forward(x)
        x = rearrange(x, "(b t) c -> b c t", b=B)
        return x


# + tags=[]
model = CLIP_ImageEncoder(backbone="ViT-B/16")
