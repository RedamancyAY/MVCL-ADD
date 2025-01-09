from math import ceil, floor, log2
import numpy as np

from .winfuns import winfuns


def nsgcqwin(*args):
    fmin, fmax, bins, sr, Ls = args[:5]

    bwfac = 1
    min_win = 4
    fractional = 0
    winfun = "hann"
    gamma = 0

    nargin = len(args)

    if nargin < 5:
        raise ValueError("Not enough input arguments")

    if nargin >= 6:
        varargin = args[5:]
        Lvar = len(varargin)
        if Lvar % 2:
            raise ValueError("Invalid input arguments")
        for i in range(0, Lvar, 2):
            if not isinstance(varargin[i], str):
                raise ValueError("Invalid input arguments")
            if varargin[i] == "min_win":
                min_win = varargin[i + 1]
            elif varargin[i] == "gamma":
                gamma = varargin[i + 1]
            elif varargin[i] == "bwfac":
                bwfac = varargin[i + 1]
            elif varargin[i] == "fractional":
                fractional = varargin[i + 1]
            elif varargin[i] == "winfun":
                winfun = varargin[i + 1]
            else:
                raise ValueError("Invalid input arguments " + varargin[i])

    nf = sr // 2

    if fmax > nf:
        fmax = nf

    fftres = sr // Ls
    b = floor(bins * log2(fmax / fmin))
    fbas = fmin * pow(2, np.arange(b).T / bins)

    Q = pow(2, 1 / bins) - pow(2, -1 / bins)
    cqtbw = Q * fbas + gamma
    cqtbw = cqtbw[:]

    # Make sure the support of highest filter won't exceed nf
    tmpIdx = np.where(fbas + cqtbw / 2 > nf)[0]
    if not np.all(tmpIdx == 0):
        fbas = fbas[: tmpIdx[0]]
        cqtbw = cqtbw[: tmpIdx[0]]

    # Make sure the support of lowest filter won't exceed DC
    tmpIdx = np.where(fbas - cqtbw / 2 < 0)[0]
    if not np.all(tmpIdx == 0):
        fbas = fbas[tmpIdx:]
        cqtbw = cqtbw[tmpIdx:]

    Lfbas = len(fbas)
    fbas = np.insert(fbas, 0, 0)
    fbas = np.insert(fbas, len(fbas), nf)
    fbas = np.insert(fbas, len(fbas), sr - fbas[Lfbas:0:-1])

    bw = cqtbw[::-1]
    bw = np.insert(bw, 0, fbas[Lfbas + 2] - fbas[Lfbas])
    bw = np.insert(bw, 0, cqtbw)
    bw = np.insert(bw, 0, 2 * fmin)

    fftres = sr / Ls
    bw = bw / fftres
    fbas = fbas / fftres

    # Centre positions of filters in DFT frame
    posit = np.zeros(fbas.shape)
    posit[: Lfbas + 2] = np.array(
        [floor(fbas_num) for fbas_num in list(fbas[: Lfbas + 2])]
    )
    posit[Lfbas + 2 :] = np.array(
        [ceil(fbas_num) for fbas_num in list(fbas[Lfbas + 2 :])]
    )

    shift = np.diff(posit)
    shift = np.insert(shift, 0, -posit[-1] % Ls)

    if fractional:
        corr_shift = fbas - posit
        M = ceil(bw + 1)
    else:
        bw = np.round(bw)
        M = bw

    for i in range(2 * (Lfbas + 1)):
        if bw[i] < min_win:
            bw[i] = min_win
            M[i] = bw[i]

    if fractional:
        temp = (
            (np.arange(ceil(M / 2) + 1) + np.arange(-floor(M / 2), 0)).conj().T
        )
        g = winfuns(winfun, (temp - corr_shift) / bw) / np.sqrt(bw)
    else:
        g = np.empty_like(bw, dtype=object)
        for i in range(len(bw)):
            g[i] = winfuns(winfun, bw[i])

    M = bwfac * np.ceil(M / bwfac)

    # Setup turnkey window for 0- and Nyquist-frequency
    for i in range(Lfbas + 2):
        if M[i] > M[i + 1]:
            start = int((np.floor(M[i] / 2) - np.floor(M[i + 1] / 2) + 1))
            end = int((np.floor(M[i] / 2) + np.ceil(M[i + 1] / 2)))
            g[i][start - 1 : end] = winfuns("hann", M[i + 1])
            g[i] = g[i] / np.sqrt(M[i])

    return g, shift, M
