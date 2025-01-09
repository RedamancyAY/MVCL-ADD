# %load_ext autoreload
# %autoreload 2

# +
import math
import random
import sys
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from ay2.tools import freeze_modules
from ay2.torch.nn import LambdaFunctionModule
from einops import rearrange

# -

try:
    from ...WaveLM.wavlm import BaseLine as WavLM
except ImportError:
    sys.path.append("../../WaveLM")
    from wavlm import BaseLine as WavLM


class WavLM_1D(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model1D = WavLM()
        self.n_dim = 768

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, 0, :]
        feature = self.model1D.pretrain_model(x)[self.model1D.pretrain_feat]  # (B, T, 768)
        return feature.mean(1)

    def compute_stage1(self, x):
        if x.ndim == 3:
            x = x[:, 0, :]
        feat = self.model1D.pretrain_model.feature_extractor(x)
        extract_features = self.model1D.pretrain_model.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)
        # 输出的extract_features其实就是输入的layer norm
        (
            hidden_states,
            extract_features,
        ) = self.model1D.pretrain_model.feature_projection(extract_features)
        return hidden_states

    def compute_transformer_layers(self, hidden_states, s, e, position_bias=None):
        # if self.position_bias is not None:
        # print(self.position_bias.shape)
        for i in range(s, e):
            layer = self.model1D.pretrain_model.encoder.layers[i]
            dropout_probability = np.random.uniform(0, 1)
            skip_the_layer = (
                self.model1D.pretrain_model.encoder.training
                and i > 0
                and (dropout_probability < self.model1D.pretrain_model.encoder.layerdrop)
            )
            if not skip_the_layer:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=None,
                    position_bias=position_bias,
                    output_attentions=False,
                    index=i,
                )
                hidden_states, position_bias = layer_outputs[:2]
        return hidden_states, position_bias

    def compute_stage2(self, hidden_states):
        position_embeddings = self.model1D.pretrain_model.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.model1D.pretrain_model.encoder.layer_norm(hidden_states)
        hidden_states = self.model1D.pretrain_model.encoder.dropout(hidden_states)
        position_bias = None
        hidden_states, position_bias = self.compute_transformer_layers(hidden_states, 0, 3, position_bias=None)
        return hidden_states, position_bias

    def compute_stage3(self, hidden_states, position_bias):
        hidden_states, position_bias = self.compute_transformer_layers(hidden_states, 3, 6, position_bias)
        return hidden_states, position_bias

    def compute_stage4(self, hidden_states, position_bias):
        hidden_states, position_bias = self.compute_transformer_layers(hidden_states, 6, 9, position_bias)
        return hidden_states, position_bias

    def compute_latent_feature(self, hidden_states, position_bias):
        hidden_states, position_bias = self.compute_transformer_layers(hidden_states, 9, 12, position_bias)  # (B, T, C)
        return hidden_states.mean(1), hidden_states


###############  Expanda module ############################


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)


class CrossAttention2D(nn.Module):
    def __init__(
        self,
        time_dim,
        spec_dim,
        feature_dim,
        num_heads=4,
        dropout_rate=0.1,
        temperature=1.0,
    ):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

        self.conv1 = nn.Conv2d(in_channels=time_dim, out_channels=feature_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=spec_dim, out_channels=feature_dim, kernel_size=1)
        self.feature_dim = feature_dim

    def forward(self, waveform, spectrogram):
        query = self.conv1(waveform).permute(0, 2, 3, 1)
        key = self.conv2(spectrogram).permute(0, 2, 3, 1)
        value = spectrogram.permute(0, 2, 3, 1)

        # print(query.shape, key.shape, value.shape, torch.matmul(query, key.transpose(-2, -1)).shape)

        attn_weights = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.feature_dim**0.5))
        out = torch.matmul(attn_weights, value).permute(0, 3, 1, 2)

        return out


class Expand(nn.Module):
    def __init__(
        self,
        time_len=149,
        time_dim=768,
        spec_height=56,
        spec_width=56,
        spec_dim=512,
        num_heads=1,
        use_PE=True,
        drop_layer=0.1,
        use_fusion=True,
    ):
        super().__init__()

        self.time_len = time_len
        self.time_dim = time_dim
        self.spec_height = spec_height
        self.spec_width = spec_width
        self.spec_dim = spec_dim
        self.use_fusion = use_fusion

        self.conv1 = nn.Conv1d(in_channels=time_len, out_channels=spec_height * spec_width, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=time_dim, out_channels=spec_dim, kernel_size=3, padding=1)

        self.layer_norm1 = nn.LayerNorm(time_dim)
        self.layer_norm2 = nn.LayerNorm(spec_dim)
        self.attn = CrossAttention2D(time_dim=time_dim, spec_dim=spec_dim, feature_dim=spec_dim)

        self.positional_encoding_wave = nn.Parameter(
            torch.randn(1, time_dim, spec_height, spec_width), requires_grad=True
        )
        self.positional_encoding_spec = nn.Parameter(
            torch.randn(1, spec_dim, spec_height, spec_width), requires_grad=True
        )
        self.use_PE = use_PE
        self.dropout = nn.Dropout(0.1)
        self.drop_layer = drop_layer

        # self.apply(init_weights)
        nn.init.xavier_uniform_(self.positional_encoding_wave)
        nn.init.xavier_uniform_(self.positional_encoding_spec)

    def compute_layernorm(self, feat, layer_norm):
        feat = rearrange(feat, "b c h w -> b h w c")
        feat = layer_norm(feat)
        feat = rearrange(feat, "b h w c -> b c h w")
        return feat

    def forward(self, x, y):
        """

        Args:
            x: (B, time_len, time_dim)
            y: (B, spec_dim, spec_height, spec_width)
        """
        if not self.use_fusion:
            return y

        if np.random.rand() < self.drop_layer:
            return y

        x = self.conv1(x)  # [B, spec_H * spec_W, time_dim]
        x = rearrange(
            x, "b (h w) c -> b c h w", h=self.spec_height, w=self.spec_width
        )  ## [B, time_dim, spec_H, spec_W]

        if self.use_PE:
            norm_x = self.compute_layernorm(x, self.layer_norm1) + self.positional_encoding_wave
            norm_y = self.compute_layernorm(y, self.layer_norm2) + self.positional_encoding_spec
        else:
            norm_x = self.compute_layernorm(x, self.layer_norm1)
            norm_y = self.compute_layernorm(y, self.layer_norm2)

        res = self.attn(norm_x, norm_y)
        res = self.dropout(res)

        res = res + y
        return res


###############  Squeeze module ############################


# class CrossAttention1D(nn.Module):
#     def __init__(self, time_dim, spec_dim, feature_dim):
#         super().__init__()
#         self.softmax = nn.Softmax(dim=-1)

#         self.linear1 = nn.Linear(time_dim, feature_dim)
#         self.linear2 = nn.Linear(spec_dim, feature_dim)
#         self.feature_dim = feature_dim

#     def forward(self, waveform, spectrogram):
#         """
#         Args:
#             waveform: (B, time_len, time_dim)
#             spectrogram: (B, time_len, spec_dim)

#         """
#         key = self.linear1(waveform)  ##  (B, time_len, feature_dim)
#         query = self.linear2(spectrogram)  ##  (B, time_len, feature_dim)
#         value = waveform

#         attn_weights = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.feature_dim**0.5))
#         out = torch.matmul(attn_weights, value)  ##  (B, time_len, feature_dim)

#         return out


# class Squeeze(nn.Module):
#     def __init__(self, time_len=149, time_dim=768, spec_height=56, spec_width=56, spec_dim=512):
#         super().__init__()

#         self.time_len = time_len
#         self.time_dim = time_dim
#         self.spec_height = spec_height
#         self.spec_width = spec_width
#         self.spec_dim = spec_dim

#         ### used to convert spec into waveform
#         self.linear = nn.Linear(spec_height * spec_width, time_len)

#         self.layer_norm1 = nn.LayerNorm(time_dim)
#         self.layer_norm2 = nn.LayerNorm(spec_dim)
#         self.attn = CrossAttention1D(time_dim=time_dim, spec_dim=spec_dim, feature_dim=spec_dim)

#     def forward(self, x, y):
#         y = rearrange(y, "b c h w -> b c (h w)")
#         y = self.linear(y)  ### # [B, time_len, spec_dim]
#         y = rearrange(y, "b c l -> b l c")


#         res = self.attn(self.layer_norm1(x), self.layer_norm2(y)) + x
#         return res


class MultiHeadCrossAttention1D(nn.Module):
    def __init__(
        self,
        time_dim,
        spec_dim,
        feature_dim,
        num_heads=4,
        dropout_rate=0.1,
        temperature=1.0,
    ):
        super().__init__()
        assert feature_dim % num_heads == 0, "Feature dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.feature_dim_head = feature_dim // num_heads
        self.temperature = temperature

        self.key_linear = nn.Linear(time_dim, feature_dim)
        self.query_linear = nn.Linear(spec_dim, feature_dim)
        self.value_linear = nn.Linear(time_dim, feature_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, waveform, spectrogram):
        batch_size = waveform.shape[0]

        # Apply linear transformations
        keys = self.key_linear(waveform)
        queries = self.query_linear(spectrogram)
        values = self.value_linear(waveform)

        # Split the last dimension into (heads, depth)
        keys = keys.view(batch_size, -1, self.num_heads, self.feature_dim_head).permute(0, 2, 1, 3)
        queries = queries.view(batch_size, -1, self.num_heads, self.feature_dim_head).permute(0, 2, 1, 3)
        values = values.view(batch_size, -1, self.num_heads, self.feature_dim_head).permute(0, 2, 1, 3)

        keys = self.dropout(keys)
        queries = self.dropout(queries)

        # Perform scaled dot-product attention
        score = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.feature_dim_head)
        attn_weights = self.softmax(score)
        out = torch.matmul(attn_weights, values).permute(0, 2, 1, 3)

        # Concatenate heads back to single head dimension
        out = out.contiguous().view(batch_size, -1, self.feature_dim)

        return out


class Squeeze(nn.Module):
    def __init__(
        self,
        time_len=149,
        time_dim=768,
        spec_height=56,
        spec_width=56,
        spec_dim=512,
        dropout_rate=0.1,
        num_heads=1,
        use_PE=True,
        drop_layer=0.1,
        use_fusion=True,
    ):
        super().__init__()

        self.linear = nn.Linear(spec_height * spec_width, time_len)
        self.layer_norm1 = nn.LayerNorm(time_dim)
        self.layer_norm2 = nn.LayerNorm(spec_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Add placeholders for positional encoding
        self.positional_encoding_wave = nn.Parameter(torch.randn(1, time_len, time_dim), requires_grad=True)
        self.positional_encoding_spec = nn.Parameter(torch.randn(1, time_len, spec_dim), requires_grad=True)
        self.use_PE = use_PE
        self.drop_layer = drop_layer
        self.use_fusion = use_fusion

        self.attn = MultiHeadCrossAttention1D(
            time_dim=time_dim,
            spec_dim=spec_dim,
            feature_dim=time_dim,
            dropout_rate=dropout_rate,
            num_heads=num_heads,
        )

        # self.apply(init_weights)
        nn.init.xavier_uniform_(self.positional_encoding_wave)
        nn.init.xavier_uniform_(self.positional_encoding_spec)

    def forward(self, x, y):
        if not self.use_fusion:
            return x

        if np.random.rand() < self.drop_layer:
            return x

        y = rearrange(y, "b c h w -> b c (h w)")
        y = self.linear(y)
        y = rearrange(y, "b c l -> b l c").contiguous()

        # Apply positional encodings
        if self.use_PE:
            norm_x = self.layer_norm1(x) + self.positional_encoding_wave
            norm_y = self.layer_norm2(y) + self.positional_encoding_spec
        else:
            norm_x = self.layer_norm1(x)
            norm_y = self.layer_norm2(y)

        res = self.attn(norm_x, norm_y)
        res = self.dropout(res)
        res += x  # Residual connection

        return res


class WaveformToSpectrogram(nn.Module):
    def __init__(self):
        super(WaveformToSpectrogram, self).__init__()
        self.conv1 = nn.Conv1d(149, 257, kernel_size=3, stride=1, padding=1)  # Adjust the channels
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(149, 257, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(257, 257, kernel_size=3, stride=1, padding=1),
        # )  # Adjust the channels

        # self.pool = nn.Linear(768, 257)

        self.pool = nn.Sequential(
            nn.Linear(768, 257),
            nn.ReLU(),
            nn.Linear(257, 257),
        )

    def __call__(self, x):
        """
        Args:
            x: (B, 149, 758)

        Returns:
            (B, 257, 257)
        """
        x = self.conv1(x)  # x: [B, 224, 768]
        x = self.pool(x)  # x: [B, 224, 224]
        return x.transpose(1, 2)


class TimeFrequencyReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = WaveformToSpectrogram()
        self.loss = nn.MSELoss()

    def __call__(self, x, y):
        x = self.module(x)
        loss = self.loss(x, y[:, 0, :, :])
        return loss


########################################################################################################################
# ###                 GatedFusionLayer
# #########################################################################################################################
class GatedFusionLayer(nn.Module):
    def __init__(self, waveform_dim, spectrogram_dim, combined_dim):
        super(GatedFusionLayer, self).__init__()

        self.proj = nn.Linear(waveform_dim, spectrogram_dim)

        self.fc = nn.Linear(spectrogram_dim * 2, combined_dim)
        self.gate_fc = nn.Linear(combined_dim, 2)

    def forward(self, waveform_features, spectrogram_features):
        waveform_features = self.proj(waveform_features)
        # Flatten features if needed
        waveform_features_flat = waveform_features.view(waveform_features.size(0), -1)
        spectrogram_features_flat = spectrogram_features.view(spectrogram_features.size(0), -1)

        # Concatenate features
        combined_features = torch.cat([waveform_features_flat, spectrogram_features_flat], dim=-1)

        # Pass through fully connected layer
        combined_features = torch.relu(self.fc(combined_features))

        # Compute gate weights
        gate = torch.sigmoid(self.gate_fc(combined_features))

        # Apply weights to the features
        weight_waveform, weight_spectrogram = gate[:, 0], gate[:, 1]
        weight_waveform = weight_waveform.view(-1, 1)
        weight_spectrogram = weight_spectrogram.view(-1, 1)

        weighted_waveform_features = waveform_features * weight_waveform
        weighted_spectrogram_features = spectrogram_features * weight_spectrogram

        # Fuse the features
        fused_features = weighted_waveform_features + weighted_spectrogram_features

        return fused_features
