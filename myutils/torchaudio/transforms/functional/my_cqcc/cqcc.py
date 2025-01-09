from .CQCC import cqcc as cqcc_func
import numpy as np

def extract_cqcc(
    x: np.ndarray, sr=16000, B=96, d=16, cf=19, ZsdD="ZsdD"
) -> np.ndarray:
    """

    Args:
        x: shape of (audio_len, audio_channels)

    
    """
    fmax=sr / 2
    fmin=fmax // (2**9)
    cqcc = cqcc_func.cqcc(x, sr, B, fmax, fmin, d, cf, ZsdD)
    return cqcc
