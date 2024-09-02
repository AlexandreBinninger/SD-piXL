import os
import numpy as np
from typing import List

def init_folders(*folders):
    for f in folders:
        dir_name = os.path.dirname(f)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

def add_suffix(path: str, suffix: str) -> str:
    if len(path) < len(suffix) or path[-len(suffix):] != suffix:
        path = f'{path}{suffix}'
    return path

def path_init(suffix: str, path_arg_ind: int, is_save: bool):
    def wrapper(func):
        def do(*args, **kwargs):
            path = add_suffix(args[path_arg_ind], suffix)
            if is_save:
                init_folders(path)
            args = [args[i] if i != path_arg_ind else path for i in range(len(args))]
            return func(*args, **kwargs)
        return do
    return wrapper

@path_init(".hex", 0, False)
def load_hex(path:str) -> np.ndarray:
    with open(path, 'r') as f:
        colors = []
        for line in f:
            hex_value = line.strip()
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
            colors.append([r, g, b])
        return np.array(colors, dtype=np.uint8)


@path_init(".hex", 1, True)
def save_hex(palette : np.ndarray, path:str):
    with open(path, 'w') as f:
        for rgb in palette:
            r = f"{rgb[0]:02x}"
            g = f"{rgb[1]:02x}"
            b = f"{rgb[2]:02x}"
            f.write(r+g+b+"\n")
            
def collect(root: str, *suffix, prefix='') -> List[List[str]]:
    if os.path.isfile(root):
        folder = os.path.split(root)[0] + '/'
        extension = os.path.splitext(root)[-1]
        name = root[len(folder): -len(extension)]
        paths = [[folder, name, extension]]
    else:
        paths = []
        root = add_suffix(root, '/')
        if not os.path.isdir(root):
            print(f'Warning: trying to collect from {root} but dir isn\'t exist')
        else:
            p_len = len(prefix)
            for path, _, files in os.walk(root):
                for file in files:
                    file_name, file_extension = os.path.splitext(file)
                    p_len_ = min(p_len, len(file_name))
                    if file_extension in suffix and file_name[:p_len_] == prefix:
                        paths.append((f'{add_suffix(path, "/")}', file_name, file_extension))
            paths.sort(key=lambda x: os.path.join(x[1], x[2]))
    return paths
