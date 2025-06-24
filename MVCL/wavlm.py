# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WavLMModel


class BaseLine(nn.Module):
    def __init__(self, pretrain_feat="last_hidden_state", backend='linear', num_classes=1, **kwargs):
        super().__init__()

        assert pretrain_feat in ["last_hidden_state", "extract_features"]
        self.pretrain_feat = pretrain_feat
        # The channels of used features for the pretrained model is 512 when using
        # the 'extract_features',  but 768 when ["last_hidden_state"] is used.
        C_features = 512 if pretrain_feat == "extract_features" else 768
        
        self.pretrain_model = WavLMModel.from_pretrained(
            "microsoft/wavlm-base"
        )

        self.backend = backend
        if backend == 'resnet':
            self.backend_model = ResNet50(
                in_channels=C_features, classes=num_classes
            )
        elif backend == 'linear':
            self.pooler = nn.AdaptiveAvgPool1d(1)
            self.backend_model = nn.Linear(C_features, num_classes)

    def get_feature(self, x):
        feature = self.pretrain_model(x)[self.pretrain_feat]
        feature = torch.transpose(feature, 1, 2)
        if self.backend == 'linear':
            feature = torch.squeeze(self.pooler(feature), -1)
        return feature
    
    def forward(self, x):
        feature = self.pretrain_model(x)[self.pretrain_feat]
        feature = torch.transpose(feature, 1, 2)
        if self.backend == 'linear':
            feature = torch.squeeze(self.pooler(feature), -1)
        # print(feature.shape, self.pooler(feature).shape)
        outputs = self.backend_model(feature)
        return outputs

    def extract_feature(self, x):
        return self.get_feature(x)

    def make_prediction(self, feature):
        outputs = self.backend_model(feature)
        return outputs

# + tags=["active-ipynb", "style-student"]
# x = torch.rand(10, 69000)
# model = BaseLine(backend='linear', pretrain_feat="last_hidden_state")
#
# model(x)
