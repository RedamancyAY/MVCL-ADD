import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from torchaudio.transforms import LFCC

def func(n):
    """求阶乘"""
    if n == 0 or n == 1:
        return 1
    else:
        return n * func(n - 1)


def Permutation_Entropy_torch(x, m, t):
    """计算排列熵值"""
    length = len(x) - (m - 1) * t
    # 重构 k*m 矩阵

    index = torch.arange(length)
    y = [x[index + i * 2] for i in range(10)]
    y = torch.stack(y, 0).transpose(0, 1)

    # 将各个分量升序排序
    S = torch.argsort(y, 1)
    values = torch.tensor([10**i for i in range(10)]).to(x.device)
    S = S * values[None, :]
    S = S.sum(-1)
    _, counts = torch.unique(S, return_counts=True)
    freq_list = counts / len(S)

    pe = torch.sum(-1 * freq_list * torch.log(freq_list))
    return pe / np.log(func(m))


def MSE_torch(signal, max_scale: int = 20):
    result = []
    length = len(signal)
    std = torch.std(signal)
    for scale in range(1, max_scale + 1):
        length_scale = length % scale
        signal_scale = signal[: length - length_scale].reshape(-1, scale)
        signal_new = torch.mean(signal_scale, dim=1)
        result.append(Permutation_Entropy_torch(signal_new, 10, 2))
    return result


def compute_mpe(x):
    signal_flu = torch.diff(x)
    scale = 20
    mpe = MSE_torch(signal_flu, scale)  # (scale,)
    mpe = torch.tensor(mpe)
    return mpe


class MPE_LFCC:
    def __init__(self):
        self.lfcc = LFCC(
            n_lfcc=60,
            speckwargs={"n_fft": 400, "hop_length": 160, "center": False},
        )

    def __call__(self, x):
        # lfcc = self.lfcc(x) # (1, h, w)
        # mpe = compute_mpe(x[0]) # (20)
        # lfcc = rearrange(lfcc, "c h w -> (c h w)")
        # res = torch.concat([mpe, lfcc])
        # return res[None, :]
        mpe = compute_mpe(x[0]) # (20)
        res = torch.concat([x, mpe[None, :]], 1)
        return res
