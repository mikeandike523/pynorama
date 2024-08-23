import os
import shutil

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from lib.image_processing import RGBAImage

this_dir = os.path.dirname(os.path.abspath(__file__))

files_dir = os.path.join(this_dir, "resources", "test-sets")

out_files_dir = os.path.join(this_dir, "resources","test-sets-background-removed")

if os.path.exists(out_files_dir):
    shutil.rmtree(out_files_dir)
os.makedirs(out_files_dir)

for dn in os.listdir(files_dir):
    os.makedirs(os.path.join(out_files_dir, dn))
    for image_file in os.listdir(os.path.join(files_dir, dn)):
        image = RGBAImage.from_file(os.path.join(files_dir, dn, image_file))
        output_file = os.path.join(out_files_dir, dn, image_file)
        image = image.remove_background_otsu(1,5,3)
        Image.fromarray(image.pixels).save(output_file)
        print(f"Removed background from {dn}/{image_file} to {output_file}")
