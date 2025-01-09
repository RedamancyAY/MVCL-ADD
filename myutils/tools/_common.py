import os
import re
import datetime
import shutil
import time

__all__ = [
    
]

def summary_torch_model(model):
    import pandas as pd
    from tabulate import tabulate

    def get_params(_model):
        total_params = sum(p.numel() for p in _model.parameters())
        total_trainable_params = sum(p.numel() for p in _model.parameters() if p.requires_grad)
        return total_params, total_trainable_params
    
    total_params, total_trainable_params = get_params(model)
    data = pd.DataFrame(columns=["Name", "Paramters", "Trainable Parameters", "Percentage%"])
    data.loc[len(data)] = ['Total', f"{total_params:,d}", f"{total_trainable_params:,d}", 100]
    
    for name, m in model.named_children():
        _total_params = sum(p.numel() for p in m.parameters())
        _total_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        data.loc[len(data)] = [name, f"{_total_params:,d}", f"{_total_trainable_params:,d}", f"{_total_params/total_params*100:.2f}"]
    print(tabulate(data, headers = 'keys', tablefmt = 'pretty', stralign=u'right'))

 
def find_unsame_name_for_file(path):
    directory, file_name = os.path.split(path)
    while os.path.isfile(path):
        pattern = '(\d+)\)\.'
        if re.search(pattern, file_name) is None:
            file_name = file_name.replace('.', '(0).')
        else:
            current_number = int(re.findall(pattern, file_name)[-1])
            new_number = current_number + 1
            file_name = file_name.replace(f'({current_number}).', f'({new_number}).')
        path = os.path.join(directory + os.sep + file_name)
    return path



def to_list(x):
    """
    if the input is not a list, return [input]
    """
    if isinstance(x, list):
        return x
    else:
        return [x]


def check_dir(file):
    """
    check the folder of file. If its folder doesn't exists, create it.
    """
    dirs = os.path.split(file)[0]
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def read_file_paths_from_folder(folder, exts):
    """
    read all file paths iteratively from a folder giving the exts.
    """
    exts = to_list(exts)

    paths = []
    for path, dir_list, file_list in os.walk(folder):
        for file_name in file_list:
            for ext in exts:
                if file_name.endswith(ext):
                    paths.append(os.path.join(path, file_name))
                    break
    return sorted(paths)


def color_print(*args):
    """
    print string with colorful background
    """
    from rich.console import Console
    string = ' '.join([str(x) for x in args])
    Console().print(f"[on #00ff00][#ff3300]{string}[/#ff3300][/on #00ff00]")




def backup_file_with_timestamp(file, delete_org=False):
    """backup file with its modify timestamp

    Args:
        file: the path of the file
        delete_org: after backup, whethere delete the original file.

    """

    if not os.path.exists(file):
        return

    m_time = os.path.getmtime(file)
    m_time = datetime.datetime.fromtimestamp(m_time)
    m_time = m_time.strftime("%Y-%m-%d-%H:%M:%S")

    if os.path.exists(file):
        filename, ext = os.path.splitext(file)
        backup_file = file.replace(ext, f"-{m_time}{ext}")
        if not os.path.exists(backup_file):
            shutil.copy2(file, backup_file)
            
        if delete_org and os.path.exists(backup_file):
            os.remove(file)



class TimerContextManager:
    def __init__(self, theme="", debug=False):
        self.debug = debug
        self.theme = theme

        
    def __enter__(self,):
        if self.debug:
            self.start = time.time()
        return self  # 可以返回不同的对象

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.debug:
            self.end = time.time()
            self.interval = self.end - self.start
            print(f"{self.theme} 代码执行时间: {self.interval} 秒")