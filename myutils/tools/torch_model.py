import torch
from hashlib import md5


def hash_tensor(tensor):
    """calculate the md5 value of a torch tensor"""

    hash = md5(str(x).encode()).hexdigest()
    return int(hash, 16)


def hash_module(m):
    """calculate the md5 value of the whole module

    We calculate the md5 of each parameter value and summarize all the md5 values
    as the md5 of the whole module.

    """
    _hashes = []
    for k, v in dict(m.state_dict()).items():
        _h = hash_tensor(v)
        _hashes.append(_h)
        return sum(_hashes)


# +
def _set_module_trainable(module, trainable: bool):
    """set `trainable` attribute for the parameters in the input module

    Args:
        module: the torch module
        trainable: True or False

    Raise:
        ValueError: raise when trainable is not True or False

    """

    if not trainable in [True, False]:
        raise ValueError(
            f"Error, please input True/False for the trainable attribute, your input is {trainable}"
        )

    for param in module.parameters():
        param.requires_grad = trainable


def freeze_modules(x):
    """
    freeze all the parameters of of x, i.e, set all parameter un-trainable.

    Args:
        x: a torch module or a list of modules
    """

    if isinstance(x, list):
        for _x in x:
            freeze_modules(_x)
    else:
        _set_module_trainable(x, False)


def unfreeze_modules(x):
    """
    unfreeze all the parameters of x, i.e, set all parameter trainable.

    Args:
        x: a torch module or a list of modules
    """
    if isinstance(x, list):
        for _x in x:
            unfreeze_modules(_x)
    else:
        _set_module_trainable(x, True)
