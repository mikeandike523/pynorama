import os
import shutil

def make_fresh_folder(folder_path):
    if os.path.exists(folder_path):
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"{folder_path} is not a directory.")
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)