import sys
import os
import shutil

from termcolor import colored

input_folder = sys.argv[1]

output_folder = sys.argv[2]

if os.path.exists(output_folder):
    if not os.path.isdir(output_folder):
        raise NotADirectoryError(f"{output_folder} is not a directory.")
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

if not os.path.exists(input_folder):
    raise FileNotFoundError("Input folder path does not exist.")
if not os.path.isdir(input_folder):
    raise ValueError("Input folder path is not a directory.")

input_folder = os.path.realpath(input_folder)
output_folder = os.path.realpath(output_folder)

discovered = []


def discover(d):
    files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    dirs = [f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]
    if any(f.endswith(".tif") and not f.startswith(".") for f in files):
        if "20x" not in d:
            discovered.append(
                os.path.relpath(
                    d,
                    input_folder,
                )
            )
    for dir in dirs:
        discover(os.path.join(d, dir))


discover(input_folder)

print(f"Discovered directories with .tif files: {'\n'.join(discovered)}")


def common_prefix(strings):
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for i, char in enumerate(shortest):
        for s in strings:
            if s[i] != char:
                return shortest[:i]
    return shortest


for d in discovered:
    test_set_name = "-".join(d.split(os.sep))
    print(colored(f"Processing {d} ({test_set_name})", "blue"))
    files = [
        f
        for f in os.listdir(os.path.join(input_folder, d))
        if f.endswith(".tif") and not f.startswith(".")
    ]
    prefix = common_prefix(files)
    files = [f[len(prefix) :] for f in files]
    min_digits = min(len(os.path.splitext(f)[0]) for f in files)
    for file in files:
        wo_ext = os.path.splitext(file)[0]
        source_name = prefix + file
        target_name = wo_ext.rjust(min_digits, "0") + ".tif"
        if not os.path.exists(os.path.join(output_folder, test_set_name)):
            os.makedirs(os.path.join(output_folder, test_set_name))
        shutil.copyfile(
            os.path.join(input_folder, d, source_name),
            os.path.join(output_folder, test_set_name, target_name),
        )
    print(files)
