from typing import List
from dataclasses import dataclass
import numpy as np
from PIL import Image
import os
import shutil


from .RGBAImage import RGBAImage


def save_image(image: RGBAImage, directory: str, filename: str) -> str:
    """Save the RGBA image as a PNG file in the specified directory."""
    img = Image.fromarray(image.pixels, "RGBA")
    file_path = os.path.join(directory, filename)
    img.save(file_path, format="PNG")
    return file_path


def create_image_arrangement(
    images: List[RGBAImage], output_folder: str
) -> str:
    # Ensure the output directory exists, recreate if necessary
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Step 1: Calculate the overall bounds
    min_x = min(image.x for image in images)
    min_y = min(image.y for image in images)
    max_x = max(image.x + image.get_width() for image in images)
    max_y = max(image.y + image.get_height() for image in images)

    width = int(max_x - min_x)
    height = int(max_y - min_y)

    # Step 2: Save images to the specified directory and reference them
    svg_images = []
    for i, image in enumerate(images):
        filename = f"image_{i}.png"
        file_path = save_image(image, output_folder, filename)
        relative_path = os.path.relpath(file_path, start=output_folder)
        svg_img_str = (
            f'<image x="{image.x - min_x}" y="{image.y - min_y}" '
            f'width="{image.get_width()}" height="{image.get_height()}" '
            f'xlink:href="{relative_path}" />'
        )
        svg_images.append(svg_img_str)

    # Step 3: Build the SVG string
    svg_str = f"""
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
    width="{width}" height="{height}">
    {''.join(svg_images)}
    </svg>
    """

    # Save the SVG string to a file
    svg_file_path = os.path.join(output_folder, "output.svg")
    with open(svg_file_path, "w") as f:
        f.write(svg_str)

    # Step 4: Create the HTML file
    html_str = (
        f"<!DOCTYPE html>"
        f'<html lang="en">'
        f"<head>"
        f'<meta charset="UTF-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        f"<title>Image Arrangement</title>"
        f"<style>"
        f"  .svg-container {{ overflow: auto; }}"
        f"</style>"
        f"</head>"
        f"<body>"
        f'<div class="svg-container">'
        f"{svg_str}"
        f"</div>"
        f"</body>"
        f"</html>"
    )

    html_file_path = os.path.join(output_folder, "arrangement.html")
    with open(html_file_path, "w") as f:
        f.write(html_str)

    return html_file_path
