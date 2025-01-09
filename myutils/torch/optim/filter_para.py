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

import torch
import torch.nn as nn


def exclude_bias_and_norm(p, *args, **kwargs):
    """
    return true if the parameters is a one-dimension tensor
    Args:
        p: a tensor
    """
    return p.ndim == 1


def filter_para_for_weight_decay(
    named_parameters, filter_func="exclude_bias_and_norm", weight_decay=0.05
):
    """
    filter parameters that using weight decay or not.

    ```python
    model = torch.nn.Module
    optim_groups = filter_para_for_weight_decay(model.named_parameters(), 
                                filter_func='exclude_bias_and_norm',
                                weight_decay=weight_decay)
    optimizer = optimizer = Adam(
            opti_groups,
            lr=self.learning_rate,
            betas=(0.5, 0.9),
        )

    ```
    Args:
        named_parameters: list of tuple (parameters, name)
        filter_func: the filter function that judge wheter a parameter uses weight decay
        weight_decay: weight decay factor

    Returns:
        a parameter group, [
            {"params": list of parameters, "weight_decay": weight_decay},
            {"params": list of parameters, "weight_decay": 0.0}
        ]
    """
    weight_decay_params, no_weight_decay_params = [], []

    if filter_func == "exclude_bias_and_norm":
        filter_func = exclude_bias_and_norm

    for pn, p in named_parameters:
        if filter_func(p, pn):
            no_weight_decay_params.append(p)
        else:
            weight_decay_params.append(p)

    optim_groups = [
        {
            "params": weight_decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_weight_decay_params,
            "weight_decay": 0.0,
        },
    ]
    return optim_groups

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# from torchvision import models
# model = models.resnet18()
#
# optim_groups = filter_para_for_weight_decay(model.named_parameters())
