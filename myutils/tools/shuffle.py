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

import numpy as np
from myutils.math import logistic_map


def random_shuffle_with_seed(_list, seed, map_func="logistic_map"):
    random_list = logistic_map(seed, len(_list))
    index = np.argsort(random_list)
    new_list = [_list[i] for i in index]

    return new_list


# + tags=["active-ipynb", "style-solution"]
# x = [1, 5, 2, 24, 11, 9]
# seed = 0.99
# random_shuffle_with_seed(x, seed)
# -

def split_list_into_chunks(_list, n_splits):
    if n_splits == 1:
        return _list
    elif n_splits < 1:
        raise ValueError("number of splits must be larger than 0!!")

    L = len(_list)
    num_per_chunk = L // n_splits
    res = []
    for i in range(n_splits):
        if i == n_splits - 1:
            _res = _list[num_per_chunk * i :]
        else:
            _res = _list[num_per_chunk * i : num_per_chunk * (i + 1)]
        res.append(_res)
    return res

# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# x = [1, 5, 2, 24, 11, 9, 7]
# split_list_into_chunks(x, 2)
