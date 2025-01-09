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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .distance import cosine_similarity


# $$
# \begin{aligned}
# \mathcal{L}_{\text {con }}= & \frac{1}{N^2} \sum_i^N\left(\sum_{j: y_i=y_j}^N\left(1-\frac{\mathbf{Z}_i \cdot \mathbf{Z}_j}{\left\|\mathbf{Z}_i\right\|\left\|\mathbf{Z}_j\right\|}\right)\right. \\
# & \left.+\sum_{j: y_i\neq y_j}^N \max \left(\frac{\mathbf{Z}_i \cdot \mathbf{Z}_j}{\left\|\mathbf{Z}_i\right\|\left\|\mathbf{Z}_j\right\|}-\alpha, 0\right)\right)
# \end{aligned}
# $$

class BinaryTokenContrastLoss(nn.Module):
    def __init__(self, alpha=0.3, distance="cosine_similarity"):
        super().__init__()
        self.alpha = alpha
        if isinstance(distance, str):
            if distance == "cosine_similarity":
                self.distance = cosine_similarity
        else:
            self.distance = distance

    def forward(self, tokens, labels):
        assert tokens.size(0) == labels.size(0)
        assert tokens.ndim == 2
        similariry_matrix = self.distance(tokens)
        label_matrix = labels[:, None] + labels[None, :]
        loss = torch.where(
            label_matrix != 1, # lable pairs (0, 0) , (1, 1)
            1 - similariry_matrix,
            torch.maximum(
                similariry_matrix - self.alpha, torch.zeros_like(similariry_matrix)
            ),
        )
        return torch.mean(loss)


class MultiClass_ContrastLoss(nn.Module):
    def __init__(self, alpha=0.3, distance="cosine_similarity"):
        super().__init__()
        self.alpha = alpha
        self.distance = distance
        if distance == "cosine_similarity":
            self.distance_func = cosine_similarity
        elif distance == "l2":
            self.distance_func = lambda x: torch.cdist(x, x)

    def forward(self, tokens, labels):
        tokens = F.normalize(tokens, dim=-1)
        assert tokens.size(0) == labels.size(0)
        assert tokens.ndim == 2
        similariry_matrix = self.distance_func(tokens)
        label_matrix = labels[:, None] - labels[None, :]
        if self.distance == "cosine_similarity":
            loss = torch.where(
                label_matrix != 0, # lable pairs (0, 0) , (1, 1)
                1 - similariry_matrix,
                torch.maximum(
                    similariry_matrix - self.alpha, torch.zeros_like(similariry_matrix)
                ),
            )
        else:
            loss = torch.where(
                label_matrix != 0, # lable pairs (0, 0) , (1, 1)
                similariry_matrix,
                torch.maximum(
                    self.alpha - similariry_matrix, torch.zeros_like(similariry_matrix)
                ),
            )
        return torch.mean(loss)


# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# from distance import cosine_similarity
# x = torch.rand(5, 128)
#
# x = F.normalize(x, dim=-1)
#
# cosine_similarity(x), torch.cdist(x[None], x[None])

# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# module1 = MultiClass_ContrastLoss(distance='l2')
# module2 = MultiClass_ContrastLoss(distance='cosine_similarity')
# module1(x, labels), module2(x, labels)
# -

class CLIPLoss1D(nn.Module):

    '''
        loss function used in the CLIP model that judges whether the (image, text)
        embedding pairs are actually pairs.
        Copy and modified from https://github.com/descriptinc/lyrebird-wav2clip/blob/1864b3924be5a785e2d49d975b8a26ff93f62951/wav2clip/pre_training/loss.py#L9
        The original version of CLIP is in lines: https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L358
    '''
    
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_image = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()
    
    
    def forward(self, image_features, text_features):
        '''
        Args:
            image_features: a tensor of shape (B, C)
            text_features: a tensor of shape (B, C)

        Returns:
            a loss
        '''
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logit_scale * text_features @ image_features.t()
        logits_per_text = logits_per_image.t() # only change this line
        
        batch_size = image_features.shape[0]
        ground_truth = torch.arange(
            batch_size, dtype=torch.long, device=image_features.device
        )
        return (
            self.loss_image(logits_per_image, ground_truth)
            + self.loss_text(logits_per_text, ground_truth)
        ) / 2
