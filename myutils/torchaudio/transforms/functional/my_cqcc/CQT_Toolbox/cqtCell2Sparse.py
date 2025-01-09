from math import log2
import numpy as np
from .cell2mat import cell2mat


def cqtCell2Sparse(c, M):
    bins = M.shape[0] // 2 - 1
    spLen = M[bins]
    cSparse = np.zeros((bins, spLen))

    M = M[: bins + 1]
    step = 1
    distinctHops = log2(M[bins] / M[1]) + 1
    curNumCoef = M[bins]

    for i in range(distinctHops):
        idx = [M == curNumCoef] + [False]
        temp = cell2mat(c[idx - 1].T)
        idx += list(range(0, len(cSparse), step))
        cSparse[idx] = temp
        step = step * 2
        curNumCoef = curNumCoef / 2

    return sparse(cSparse)


def sparse(m):
    index_res = np.where(m > 0)
    # index_list = [index for index in index_res]
    value_list = [m[index[0]][index[1]] for index in index_res]
    return value_list
