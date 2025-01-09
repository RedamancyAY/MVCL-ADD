import importlib


# In dict, `key` is the name of cls, function or module
#  `value` is its path, where "/" denotes it is a cls or function in a py file (module).
__all__ = {
    "Xception": "_xception/Xception",
    "XceptionClassification": "_xception/XceptionClassification",
    "Resnet18FeatureExtractor": "resnet/Resnet18FeatureExtractor",
    'EfficientNetB4' : '_efficientNet/EfficientNetB4'
}


def __getattr__(name):
    if name in __all__.keys():
        path = __all__[name]

        if not "/" in path:
            return importlib.import_module("." + path, __name__)
        else:
            py_path = path.split("/")[0]
            cls_name = path.split("/")[1]
            module = importlib.import_module("." + py_path, __name__)
            module = getattr(module, cls_name)
            return module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__.keys()
