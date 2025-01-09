"""Helper for generating derivatives"""

import numpy as np
import scipy


def Deltas(x, hlen):
    """Delta and acceleration coefficients

    Parameters
    ----------

    x : ndarray
        input signal
    hlen : int
        length of the delta coefficients

    Returns
    -------

    D : ndarray
        delta coefficients

    References
    ----------

    .. [1] Young S.J., Evermann G., Gales M.J.F., Kershaw D., Liu X., Moore G., Odell J., Ollason D.,
       Povey D., Valtchev V. and Woodland P., The HTK Book (for HTK Version 3.4) December 2006.

    """
    win = list(range(hlen, -hlen - 1, -1))

    xx_1 = np.tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = np.tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T

    xx = np.concatenate([xx_1, x, xx_2], axis=-1)

    D = scipy.signal.lfilter(win, 1, xx)

    D = D[:, hlen * 2 :]
    D = D / (2 * sum(np.arange(1, hlen + 1)) ** 2)

    return D
