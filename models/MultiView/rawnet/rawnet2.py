# %load_ext autoreload
# %autoreload 2

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torchaudio
from copy import deepcopy
import torch.nn.functional as F

try:
    from .v2 import RawNet2 as org_model
except ImportError:
    from v2 import RawNet2 as org_model
    

# RAW_NET2_CONFIG = {
#     "nb_samp": 48000,
#     "first_conv": 1024,  # no. of filter coefficients
#     "in_channels": 1,  # no. of filters channel in residual blocks
#     "filts": [20, [20, 20], [20, 128], [128, 128]],
#     "blocks": [2, 4],
#     "nb_fc_node": 1024,
#     "gru_node": 1024,
#     "nb_gru_layer": 3,
#     "nb_classes": 1,
# }


RAW_NET2_CONFIG = {
    "nb_samp": 48000,
    "first_conv": 1024,  # no. of filter coefficients
    "in_channels": 1,  # no. of filters channel in residual blocks
    # "filts": [20, [20, 20], [20, 128], [128, 128]],
    "filts": [64, [64, 128], [128, 256], [256, 512]],
    "blocks": [2, 4],
    "nb_fc_node": 512,
    "gru_node": 512,
    "nb_gru_layer": 3,
    "nb_classes": 1,
}



# def RawNet2():
#     return org_model(deepcopy(RAW_NET2_CONFIG))


class RawNet2(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            cfg = RAW_NET2_CONFIG
        self.model = org_model(deepcopy(cfg))
        self.verbose = 0
    
    def preprocess(self, x, stage='test'):
        x = x[:, 0, :]
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = self.model.ln(x)
        x=x.view(nb_samp,1,len_seq)
        return x    

    def compute_stage1(self, x):
        x = self.preprocess(x)
        x = torch.abs(self.model.first_conv(x))
        if self.verbose:
            print("Stage 1, sin conv : ", x.shape)
        x = F.max_pool1d(x, 4)
        if self.verbose:
            print("Stage 1, pool : ", x.shape)
        x = self.model.first_bn(x)
        x = self.model.lrelu_keras(x)
        
        # x = self.model.block1(x)
        if self.verbose:
            print("Stage 1: ", x.shape)
        return x

    def compute_stage2(self, x):
        x = self.model.block0(x)
        if self.verbose:
            print("Stage 2: ", x.shape)
        x = self.model.block1(x)
        if self.verbose:
            print("Stage 2: ", x.shape)
        return x

    def compute_stage3(self, x):
        x = self.model.block2(x)
        x = self.model.block3(x)
        if self.verbose:
            print("Stage 3: ", x.shape)
        return x

    def compute_stage4(self, x):
        x = self.model.block4(x)
        x = self.model.block5(x)
        if self.verbose:
            print("Stage 4: ", x.shape)
        return x
    
    
    def get_hidden_state(self, x):
        x = F.max_pool1d(torch.abs(self.first_conv(x)), 3)
        x = self.first_bn(x)
        x = self.lrelu_keras(x)

        
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.bn_before_gru(x)
        x = self.lrelu_keras(x)

        return x


    def extract_feature(self, x):
        x = self.get_hidden_state(x)
        x = self.get_final_feature(x)
        return x
    

    def compute_latent_feature(self, x):
        return self.get_final_feature(x)
    
    def get_final_feature(self, x):
        x = self.model.bn_before_gru(x)
        x = self.model.lrelu_keras(x)

        x = x.permute(0, 2, 1)  #(batch, filt, time) >> (batch, time, filt)
        self.model.gru.flatten_parameters()
        x, _ = self.model.gru(x)
        x = x[:,-1,:]
        
        code = self.model.fc1_gru(x)
        code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
        code = torch.div(code, code_norm)
        
        return code




# +
# model = RawNet2(deepcopy(RAW_NET2_CONFIG))

# x = torch.randn(3, 48000)
# model(x)
