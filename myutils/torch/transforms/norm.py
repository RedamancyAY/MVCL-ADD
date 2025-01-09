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

# + tags=[]
import torch
import torchvision


# + tags=[]
class Normalize(object):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format="NCHW"):
        self.t = torchvision.transforms.Normalize(mean=mean, std=std)
        self.data_format = data_format

    def __call__(self, x, *kargs, **kwargs):
        if self.data_format.endswith("CHW"):
            return self.t(x)
        if self.data_format == "NCTHW":
            x = torch.transpose(x, 1, 2)
            x = self.t(x)
            x = torch.transpose(x, 1, 2)
            return x
        raise ValueError(self.data_format, "wrong data format")

# + tags=["active-ipynb"]
# x = torch.rand(5, 30, 3, 224, 224)
# y = torch.rand(5, 3, 10, 224, 224)
# Normalize()(x, y).shape, Normalize(data_format="NCTHW")(y).shape
