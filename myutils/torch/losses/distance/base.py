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

import torch
import torch.nn.functional as F


def l2_normalize(x, dim=None):
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / norm


def cosine_similarity(input1, input2=None):
    """calculate cosine_similarity among N vectors

    Args:
        input1: (N, L)
        input2: (N, L) or None. when `input2` is None, input2 will be input1

    Return:
        similarity matrix `C` with size $N \times N$, where `C_ij` is the
        cosine_similarity between input1[i, :] and input[j, :]
    """
    assert input1.ndim == 2
    if input2 is not None:
        assert input2.ndim == 2
    input1 = l2_normalize(input1, dim=-1)
    input2 = l2_normalize(input2, dim=-1) if input2 is not None else input1
    return torch.matmul(input1, input2.t())
