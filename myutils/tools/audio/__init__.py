import importlib


# In dict, `key` is the name of cls, function or module
#  `value` is its path, where "/" denotes it is a cls or function in a py file (module).
__all__ = {
    "AudioConverter": "converter/AudioConverter",
    "read_audio_fps_len": "read_info/read_audio_fps_len",
    "read_audio_fps": "read_info/read_audio_fps_len",
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
