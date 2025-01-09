# +
from torch import nn
import torch
from pytorch_wavelets import DWTForward
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math

from efficientnet_pytorch import EfficientNet


class EfficientNetB4(nn.Module):
    def __init__(self, model: str = 'efficientnet-b4'):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained(model, advprop=True)
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 2)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        return x

    def conv_features(self, x):
        return self.efficientnet.extract_features(x)


# +
# efficientnet = EfficientNet.from_pretrained('efficientnet-b4')


# x = torch.randn(2, 3, 299, 299)
# efficientnet.extract_features(x).shape
