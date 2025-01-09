import numpy as np
import torch
import torchaudio

try:
    from .functional import extract_cqcc
except ImportError:
    from .functional import extract_cqcc


# 来自：[Py-CQCC/CQT_Toolbox/nsgtf_real.py at master · ShubhankarKG/Py-CQCC · GitHub](https://github.com/ShubhankarKG/Py-CQCC)

class CQCC:
    def __init__(
        self,
    ):
        pass
    
    def __call__(self, wave):
        dtype = type(wave)

        if dtype is torch.Tensor:
            wave = wave.numpy()

        if wave.shape[0] == 1:
            wave = np.transpose(wave, (0, 1))
        
        spec = extract_cqcc(wave)
        spec = spec.astype(np.float32)

        if dtype is torch.Tensor:
            return torch.from_numpy(spec)
        return spec

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# // x = torch.randn(13,1, 48000)
# // module = LFCC()
# // print(module(x).shape)
