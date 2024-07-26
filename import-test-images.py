import os
import shutil


def clear_folder(path):
    items = list(os.listdir(path))
    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)


def select_and_copy(A, B):
    B = os.path.realpath(B)
    clear_folder(B)

    items = list(os.listdir(A))
    items = list(filter(lambda p: p.endswith(".tif"), items))

    print(f"Found {len(items)} images in {A}")

    for item in items:
        print(f"Copying {item}...")
        full_path = os.path.join(A, item)
        basename_no_ext = os.path.splitext(item)[0]
        split_name = basename_no_ext.split("_")
        if len(split_name) < 2:
            raise Exception(f"Unexpected format for {full_path}: {split_name}")
        last_section = split_name[-1]
        stripped = last_section.lstrip("0")
        if len(stripped) == 0:
            stripped = "0"
        as_number = int(stripped)
        target_name = f"{as_number}.tif"
        target_name = os.path.join(B, target_name)
        shutil.copyfile(full_path, target_name)

    print(f"Copied {len(items)} images to {B}")


def main():
    mapping = {
        "/Users/michaelsohnen/Downloads/1": "./testing/resources/test-set-1",
        "/Users/michaelsohnen/Downloads/7": "./testing/resources/test-set-2",
    }
    for A, B in mapping.items():
        select_and_copy(A, B)


if __name__ == "__main__":
    main()
