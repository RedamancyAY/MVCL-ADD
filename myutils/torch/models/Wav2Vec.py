# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=["active-ipynb"] editable=true slideshow={"slide_type": ""}
# %load_ext autoreload
# %autoreload 2

# + tags=[]
import torch
import torch.nn as nn

try:
    from transformers import AutoProcessor, Wav2Vec2Model
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Please install transformers by: pip install transformers"
    )


# + tags=[]
class Wav2vec2Base(nn.Module):
    """
    The pretrained Wav2Vec2Model used for extract audio features.

    Attributes:
        model: the pretrained Wav2Vec2Model
        output_feature: the feature map extract by Wav2vecModel, either its
            'last_hidden_state' or its 'extract_features'. After looking the source code, I found that 
            the extract_features is just the tensor before the transformer. Thus, it is better to use 
            last_hidden_state.
    """

    def __init__(
        self, output_feature="last_hidden_state", pretrain_path=None, cache_dir=None
    ):
        super().__init__()

        pretrain_path = pretrain_path or "facebook/wav2vec2-base-960h"
        self.model = Wav2Vec2Model.from_pretrained(pretrain_path, cache_dir=cache_dir)
        if not output_feature in ["last_hidden_state", "extract_features"]:
            raise ValueError(
                "output feature should be `last_hidden_state` or `extract_features`"
            )
        self.output_feature = output_feature
        self.model.feature_extractor = MyWav2Vec2FeatureEncoder(self.model.feature_extractor)

    def forward(self, x):
        """
        Args:
            x: audio waveform with shape of (B, C_in, L)

        Returns:
            the feature map extract by Wav2vecModel, either its 'last_hidden_state'
            or its 'extract_features'. Shape is (B, T, C_out)
        """
        if len(x.shape) == 3:
            x = x[:, 0, :]
        outputs = self.model(x)
        return outputs[self.output_feature]

    def encode_audio(self, x):
        output = self.forward(x)
        output = torch.transpose(output, 1, 2)
        return output


# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# model = Wav2vec2Base(
#     pretrain_path="/home/ay/data/DATA/0-model_weights/models--facebook--wav2vec2-base-960h",
# )
# x = torch.randn(2, 1, 25 * 19 * 19)
# y = model(x)
# model.model.training = True
# model.model._requires_grad = True
# model.encode_audio(x).shape

# + tags=[]
class MyWav2Vec2FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""

    def __init__(self, feature_encoder):
        super().__init__()
        
        self.feature_encoder = feature_encoder
        

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        for conv_layer in self.feature_encoder.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states

# + tags=[]
# model.model.feature_extractor = MyWav2Vec2FeatureEncoder(model.model.feature_extractor)

# + tags=[]
# x = torch.randn(2, 8100)
# feat = model.model.feature_extractor(x)
# H, F = model.model.feature_projection(feat.transpose(1, 2))
# H.shape, F.shape

# x = torch.randn(2, 512, 768)
# m = model.model.encoder
# m(x)['last_hidden_state'].shape
