"""Main functions for generating CQCC"""

from math import ceil, log2
import numpy as np
import librosa
import scipy
from .delta import Deltas

from ..CQT_Toolbox.cqt import cqt


def cqcc(*args):
    """Constant Q cepstral coefficients (CQCC)

    Returns the CQCC of an audio signal

    Parameters
    ----------

    x : ndarray
        input signal
    fs : int
        sampling rate of the signal
    B : int
        number of bins per octave [default = 96]
    fmax : int
        highest frequency to be analyzed [default = Nyquist frequency]
    fmin : int
        lowest frequency to be analyzed [default = ~20Hz to fullfill an integer number of octave]
    d : int
        number of uniform samples in the first octave [default 16]
    cf : int
        number of cepstral coefficients excluding 0'th coefficient [default 19]
    ZsdD : str
        any sensible combination of the following  [default ZsdD]:
        'Z' : include 0'th order cepstral coefficient
        's' : include static coefficients (c)
        'd' : include delta coefficients (dc/dt)
        'D' : include delta-delta coefficients (d^2c/dt^2)


    Returns
    -------

    CQCC : ndarray
        constant Q cepstral coefficients (nCoeff x nFea)


    See Also
    --------

    CQCC_Toolbox.cqt : CQT

    References
    ----------

    .. [1] M. Todisco, H. Delgado, and N. Evans. A New Feature for
       Automatic Speaker Verification Anti-Spoofing: Constant Q
       Cepstral Coefficients. Proceedings of ODYSSEY - The Speaker
       and Language Recognition Workshop, 2016.

    .. [2] C. Sch�rkhuber, A. Klapuri, N. Holighaus, and M. D�fler.
       A Matlab Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy
       Transforms. Proceedings AES 53rd Conference on Semantic Audio, London,
       UK, Jan. 2014. http://www.cs.tut.fi/sgn/arg/CQT/

    .. [3] G. A. Velasco, N. Holighaus, M. D�fler, and T. Grill. Constructing an
       invertible constant-Q transform with non-stationary Gabor frames.
       Proceedings of DAFX11, Paris, 2011.

    .. [4] N. Holighaus, M. D�fler, G. Velasco, and T. Grill. A framework for
       invertible, real-time constant-q transforms. Audio, Speech, and
       Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.

    """
    nargin = len(args)

    if nargin < 2:
        raise ValueError("Not enough input arguments")

    x, fs = args[0], args[1]

    B = 96 if nargin < 3 else int(args[2])

    fmax = fs / 2 if nargin < 4 else int(args[3])

    if nargin < 5:
        oct = ceil(log2(fmax / 20))
        fmin = fmax / 2 ** oct
    else:
        fmin = int(args[4])

    # d = 16 if nargin < 6 else int(args[5])

    cf = 19 if nargin < 7 else int(args[6])

    ZsdD = "ZsdD" if nargin < 8 else args[7]

    gamma = 228.7 * (2 ** (1 / B) - 2 ** (-1 / B))

    # CQT Computing
    Xcq = cqt(x, B, fs, fmin, fmax, "rasterize", "full", "gamma", gamma)

    # Log Power Spectrum
    absCQT = np.abs(Xcq["c"])

    # TimeVec = np.arange(1, absCQT.shape[1]+1).reshape(1, -1)
    # TimeVec = TimeVec*Xcq['xlen'] / absCQT.shape[1] / fs

    # FreqVec = np.arange(0, absCQT.shape[0]).reshape(1, -1)
    # FreqVec = fmin * 2 ** (FreqVec / B)

    eps = 2.2204e-16
    LogP_absCQT = np.log(absCQT ** 2 + eps)

    # Uniform Resampling
    # kl = B * log2(1 + 1 / d)
    Ures_LogP_absCQT = librosa.resample(LogP_absCQT.T, orig_sr=fs, target_sr=9_562).T
    # Ures_FreqVec = None

    # DCT
    CQcepstrum = scipy.fftpack.dct(
        Ures_LogP_absCQT, type=2, axis=1, norm="ortho"
    )

    # Dynamic Coefficients
    if "Z" in ZsdD:
        scoeff = 1
    else:
        scoeff = 2
    CQcepstrum_temp = CQcepstrum[scoeff - 1 : cf + 1, :]
    f_d = 3

    if ZsdD.replace("Z", "") == "sdD":
        CQcc = np.concatenate(
            [
                CQcepstrum_temp,
                Deltas(CQcepstrum_temp, f_d),
                Deltas(Deltas(CQcepstrum_temp, f_d), f_d),
            ],
            axis=0,
        )
    elif ZsdD.replace("Z", "") == "sd":
        CQcc = np.concatenate(
            [CQcepstrum_temp, Deltas(CQcepstrum_temp, f_d)], axis=0
        )
    elif ZsdD.replace("Z", "") == "sD":
        CQcc = np.concatenate(
            [CQcepstrum_temp, Deltas(Deltas(CQcepstrum_temp, f_d), f_d)],
            axis=0,
        )
    elif ZsdD.replace("Z", "") == "s":
        CQcc = CQcepstrum_temp
    elif ZsdD.replace("Z", "") == "d":
        CQcc = Deltas(CQcepstrum_temp, f_d)
    elif ZsdD.replace("Z", "") == "D":
        CQcc = Deltas(Deltas(CQcepstrum_temp, f_d), f_d)
    elif ZsdD.replace("Z", "") == "dD":
        CQcc = np.concatenate(
            [
                Deltas(CQcepstrum_temp, f_d),
                Deltas(Deltas(CQcepstrum_temp, f_d), f_d),
            ],
            axis=0,
        )

    return CQcc.T
