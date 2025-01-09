from math import ceil, log2
from .cell2mat import cell2mat
from .cqtCell2Sparse import cqtCell2Sparse
from .nsgcqwin import nsgcqwin
import numpy as np

from .nsgtf_real import nsgtf_real


def cqt(*args):
    rasterize = "full"
    phasemode = "global"
    outputFormat = "sparse"
    normalize = "sine"
    windowFct = "hann"
    gamma = 0

    x, B, fs, fmin, fmax = args[:5]
    if len(args) >= 6:
        varargin = args[5:]
        Larg = len(varargin)
        for i in range(0, Larg, 2):
            if varargin[i] == "rasterize":
                rasterize = varargin[i + 1]
            elif varargin[i] == "phasemode":
                phasemode = varargin[i + 1]
            elif varargin[i] == "format":
                outputFormat = varargin[i + 1]
            elif varargin[i] == "normalize":
                normalize = varargin[i + 1]
            elif varargin[i] == "win":
                windowFct = varargin[i + 1]
            elif varargin[i] == "gamma":
                gamma = varargin[i + 1]

    # Window design
    g, shift, M = nsgcqwin(
        fmin,
        fmax,
        B,
        fs,
        len(x),
        "winfun",
        windowFct,
        "gamma",
        gamma,
        "fractional",
        0,
    )
    fbas = np.cumsum(shift[1:]) / len(x)
    fbas = fbas[0 : M.shape[0] // 2 - 1]

    # Compute coefficients
    bins = M.shape[0] // 2 - 1
    if rasterize == "full":
        M[1 : bins + 1] = M[bins + 1]
        M[bins + 2 :] = M[bins:0:-1]

    elif rasterize == "piecewise":
        temp = M[bins]
        octs = ceil(log2(fmax / fmin))
        # Make sure that the number of coefficients in the highest octave is dividable by 2 atleast octs-times
        temp = ceil(temp / 2 ** octs) * 2 ** octs
        mtemp = temp / M
        mtemp = 2 ** ceil(log2(mtemp) - 1)
        mtemp = temp / mtemp
        mtemp[bins + 1] = M[bins + 1]  # Don't rasterize Nyquist bin
        mtemp[0] = M[0]  # Don't rasterize DC bin
        M = mtemp

    if normalize in ["sine", "Sine", "SINE", "sin"]:
        normFacVec = 2 * M[: bins + 2] / len(x)
    elif normalize in ["impulse", "Impulse", "IMPULSE", "imp"]:
        normFacVec = 2 * M[bins + 2] / [len(cell) for cell in g]
    elif normalize in ["none", "None", "NONE", "no"]:
        normFacVec = np.ones((bins + 2, 1))

    normFacVec = np.append(normFacVec, normFacVec[-2:0:-1])
    g = g[: 2 * bins + 2] * normFacVec[: 2 * bins + 2]
    g = g.T

    c, _ = nsgtf_real(x, g, shift, M, phasemode)

    if rasterize == "full":
        cDC = cell2mat(c[0])
        cNyq = cell2mat(c[bins + 1])
        c = cell2mat(c[1 : bins + 1])

    elif rasterize == "piecewise":
        cDC = cell2mat(c[0])
        cNyq = cell2mat(c[bins + 1])
        if outputFormat == "sparse":
            c = cqtCell2Sparse(c, M).T
        else:
            c = c[1:-2]

    else:
        cDC = cell2mat(c[0])
        cNyq = cell2mat(c[-1])
        c = c[1:-2]

    return dict(
        c=c.T,
        g=g,
        shift=shift,
        M=M,
        xlen=len(x),
        phasemode=phasemode,
        rast=rasterize,
        fmin=fmin,
        fmax=fmax,
        B=B,
        cDC=cDC,
        cNyq=cNyq,
        format=outputFormat,
        fbas=fbas,
    )
