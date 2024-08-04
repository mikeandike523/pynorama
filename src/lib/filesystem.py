import os
import random
import shutil


def make_fresh_folder(folder_path):
    if os.path.exists(folder_path):
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"{folder_path} is not a directory.")
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def get_random_unique_in_folder(folder_path, prefix="", extension=""):
    while True:
        random_name = f"{prefix}_{random.randint(10000, 99999)}_{extension}"
        random_path = os.path.join(folder_path, random_name)
        if not os.path.exists(random_path):
            return random_path
