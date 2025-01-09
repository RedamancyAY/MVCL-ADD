import importlib


# In dict, `key` is the name of cls, function or module
#  `value` is its path, where "/" denotes it is a cls or function in a py file (module).
__all__ = {
    "InTheWild_AudioDs": "in_the_wild/InTheWild_AudioDs",
    "WaveFake_AudioDs": "wavefake/WaveFake_AudioDs",
    "DECRO_AudioDs" : "_DECRO/DECRO_AudioDs",
    "LibriSeVoc_AudioDs" : "LibriSeVoc/LibriSeVoc_AudioDs",
    "VGGSound_AudioDs" : "_VGGSound/VGGSound_AudioDs",
    "MLAAD_AudioDs" : "_MLAAD/MLAAD_AudioDs",
    "ASV2019LA_AudioDs" : "_ASV2019/ASV2019LA_AudioDs",
    "ASV2021_AudioDs" : "_ASV2021/ASV2021_AudioDs", # DF task
    "ASV2021LA_AudioDs" : "_ASV2021_LA/ASV2021LA_AudioDs", # LA task
    "Common_Voice_AudioDs" : "_common_voice/Common_Voice_AudioDs",
    "Partial_CommonVoice_AudioDs" : "_common_voice/Partial_CommonVoice_AudioDs",
    "MultiLanguageCommonVoice" : "_common_voice/MultiLanguageCommonVoice",
    "Codecfake_AudioDs" :  "_codecfake/Codecfake_AudioDs",
    "CDADD_AudioDs": "_CDADD/CDADD_AudioDs"
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
