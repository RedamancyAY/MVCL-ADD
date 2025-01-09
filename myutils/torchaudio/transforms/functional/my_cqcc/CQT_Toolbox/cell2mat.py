import numpy as np


def cell2mat(c):
    c = np.stack(c)
    if c.ndim == 3:
        c = np.squeeze(c, axis=-1)
    c = c.T
    return c
