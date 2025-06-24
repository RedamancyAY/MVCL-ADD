# %load_ext autoreload
# %autoreload 2

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torchaudio
from copy import deepcopy
import torchaudio
try:
    from .model.encoder import ResNetExtractor
except ImportError:
    from model.encoder import ResNetExtractor
    

MODEL_URL = "https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt"


def get_model(device="cpu", pretrained=True, frame_length=None, hop_length=None):
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            MODEL_URL, map_location=device, progress=True
        )
        model = ResNetExtractor(
            checkpoint=checkpoint,
            scenario="finetune",  # frozen
            transform=True,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    else:
        model = ResNetExtractor(
            scenario="supervise", frame_length=frame_length, hop_length=hop_length
        )
    model.to(device)
    return model


class ResNet(nn.Module):
    def __init__(self, verbose=0, pretrained=True):
        super().__init__()

        self.model = get_model(pretrained=pretrained).encoder

        # from torchvision.models import resnet18
        # model = resnet18(weights='DEFAULT')
        # sd = model.state_dict()
        # sd['conv1.weight'] = torch.mean(sd['conv1.weight'], dim=1, keepdims=True)
        # _ = self.model.load_state_dict(sd, strict=False)


        
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
        # self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=353) # original

        self.verbose = verbose
    
    def preprocess(self, x, stage="test"):
        # x = self.model.spectrogram(x)
        x = self.spectrogram(x)
        x = torch.log(x + 1e-7)
        # x = (x - torch.mean(x)) / (torch.std(x) + 1e-9)

        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )
        return x

    def compute_stage1(self, x, preprocess=True, first_conv=True, spec_aug=None):
        if preprocess:    
            x = self.preprocess(x)
            raw_spec = x
            if spec_aug is not None:
                # print('use spec aug', x.shape)
                x = spec_aug.batch_apply(x)
        
        if first_conv:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
        if self.verbose:
            print('ResNet Stage 1: input after first conv shape', x.shape)
        
        x = self.model.layer1(x)
        if self.verbose:
            print('ResNet Stage 1: output shape', x.shape)
        return x, raw_spec

    def compute_stage2(self, x):
        x = self.model.layer2(x)
        if self.verbose:
            print('ResNet Stage 2: output shape', x.shape)
        return x

    def compute_stage3(self, x):
        x = self.model.layer3(x)
        if self.verbose:
            print('ResNet Stage 3: output shape', x.shape)
        return x

    def compute_stage4(self, x):
        x = self.model.layer4(x)
        if self.verbose:
            print('ResNet Stage 4: output shape', x.shape)
        return x


    def compute_stage(self, x, idx):
        if idx == 2:
            return self.compute_stage2(x)
        if idx == 3:
            return self.compute_stage3(x)
        if idx == 4:
            return self.compute_stage4(x)
        
    
    def compute_latent_feature(self, x):

        # print(x.shape)
        
        x = self.model.avgpool(x) # (64, 512, 9, 9) -> (64, 512)
        x = x.reshape(x.size(0), -1)
        x = self.feature_norm(x)
        if self.verbose:
            print("ResNet Latent Feature: output shape", x.shape)
        return x

    
    def get_hidden_state(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x

    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10
        code = torch.div(code, code_norm)
        return code
        # feat = code
        # feat = feat / (1e-9 + torch.norm(feat, p=2, dim=-1, keepdim=True))
        
        # return feat
    
    def extract_feature(self, x):
        x = self.get_hidden_state(x)
        x, conv_feat = self.get_final_feature(x)
        return x

    
    
    def get_final_feature(self, x):
        conv_feat = self.model.layer4(x)
        x = self.model.avgpool(conv_feat)
        x = x.reshape(x.size(0), -1)

        # x = self.feature_norm(x)

        return x, conv_feat




def convert_2d_to_1d(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Get the parameters of the 2D convolutional layer
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            bias = module.bias is not None

            # Create a new 1D convolutional layer with equivalent parameters
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_size[0], stride[0]**2, padding[0], dilation[0], groups, bias)
            conv1d.weight.data.copy_(torch.mean(module.weight, dim=2) /kernel_size[0] )
            if module.bias is not None:
                conv1d.bias.data.copy_(module.bias)
            
            # Replace the 2D convolutional layer with the new 1D convolutional layer
            setattr(model, name, conv1d)
            
        if isinstance(module, nn.BatchNorm2d):
            # Get the parameters of the 2D BatchNorm layer
            num_features = module.num_features
            eps = module.eps
            momentum = module.momentum
            affine = module.affine
            track_running_stats = module.track_running_stats

            # Create a new 1D BatchNorm layer with equivalent parameters
            bn1d = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

            bn1d.weight.data.copy_(module.weight)
            bn1d.bias.data.copy_(module.bias)
            bn1d.running_mean.data.copy_(module.running_mean)
            bn1d.running_var.data.copy_(module.running_var)

            
            # Replace the 2D BatchNorm layer with the new 1D BatchNorm layer
            setattr(model, name, bn1d)

        if isinstance(module, nn.MaxPool2d):
            # Get the parameters of the 2D MaxPool layer
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            return_indices = module.return_indices
            ceil_mode = module.ceil_mode

            # Create a new 1D MaxPool layer with equivalent parameters
            pool1d = nn.MaxPool1d(kernel_size, stride**2, padding, dilation, return_indices, ceil_mode)

            # Replace the 2D MaxPool layer with the new 1D MaxPool layer
            setattr(model, name, pool1d)

        if isinstance(module, nn.AdaptiveAvgPool2d):
            pool1d = nn.AdaptiveAvgPool1d(1)
            setattr(model, name, pool1d)
            

        # Recursively call the function for nested modules
        elif isinstance(module, nn.Module):
            convert_2d_to_1d(module)

