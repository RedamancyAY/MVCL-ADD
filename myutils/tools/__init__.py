# from ._common import to_list, check_dir, read_file_paths_from_folder

# from .progress_bar import rich_bar


import importlib


# In dict, `key` is the name of cls, function or module
#  `value` is its path, where "/" denotes it is a cls or function in a py file (module).
__all__ = {
    "to_list": "_common/to_list",
    "check_dir": "_common/check_dir",
    "color_print": "_common/color_print",
    "summary_torch_model": "_common/summary_torch_model",
    "find_unsame_name_for_file": "_common/find_unsame_name_for_file",
    "read_file_paths_from_folder": "_common/read_file_paths_from_folder",
    "backup_file_with_timestamp":"_common/backup_file_with_timestamp",
    "TimerContextManager" : "_common/TimerContextManager",
    #-------
    "rich_bar": "progress_bar/rich_bar",
    "split_list_into_chunks": "shuffle/split_list_into_chunks",
    "random_shuffle_with_seed": "shuffle/random_shuffle_with_seed",
    #----
    "freeze_modules": "torch_model/freeze_modules",
    "unfreeze_modules": "torch_model/unfreeze_modules",
    "hash_module":"torch_model/hash_module",
    "hash_tensor":"torch_model/hash_tensor",
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
