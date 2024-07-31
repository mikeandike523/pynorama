from typing import List
import numpy as np
from PIL import Image
import os
import shutil

from .RGBAInfiniteCanvas import RGBAInfiniteCanvas


def save_image(image: np.ndarray, directory: str, filename: str) -> str:
    """Save the RGBA image as a PNG file in the specified directory."""
    img = Image.fromarray(image, "RGBA")
    file_path = os.path.join(directory, filename)
    img.save(file_path, format="PNG")
    return file_path


def create_image_arrangement(
    images: List[np.ndarray], locations: List[np.ndarray], output_file: str
) -> str:
    
    locations = np.array(locations,float).round().astype(int)

    tlc = np.min(locations,axis=0)

    locations = locations - tlc



    # Ensure the output directory exists, recreate if necessary
    if os.path.exists(output_file):
        os.remove(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Step 1: Calculate the overall bounds
    min_x = min(location[0] for location in locations)
    min_y = min(location[1] for location in locations)
    max_x = max(
        location[0] + image.shape[1] for location, image in zip(locations, images)
    )
    max_y = max(
        location[1] + image.shape[0] for location, image in zip(locations, images)
    )

    width = int(max_x - min_x)
    height = int(max_y - min_y)

    infinite_canvas = RGBAInfiniteCanvas(width, height)

    for i, [image, location] in enumerate(zip(images, locations)):
        print(f"Processing image {i+1}/{len(images)}...")
        infinite_canvas.place_pixel_array(
            image, *list(np.round(np.array(location)).astype(int))
        )
        print(
            f"Infinite Canvas Memory Footprint: {infinite_canvas.get_footprint_human_readable()}"
        )

    infinite_canvas.save_canvas(output_file)

    # # Step 2: Save images to the specified directory and reference them
    # svg_images = []
    # for i, [image, location] in enumerate(zip(images, locations)):
    #     print(f"Processing image {i+1}/{len(images)}...")
    #     filename = f"image_{i}.png"
    #     file_path = save_image(image, output_folder, filename)
    #     relative_path = os.path.relpath(file_path, start=output_folder)
    #     image_x = location[0]
    #     image_y = location[1]
    #     svg_img_str = (
    #         f'<image x="{image_x - min_x}" y="{image_y - min_y}" '
    #         f'width="{image.shape[1]}" height="{image.shape[0]}" '
    #         f'xlink:href="{relative_path}" />'
    #     )
    #     svg_images.append(svg_img_str)

    # # Step 3: Build the SVG string
    # svg_str = f"""
    # <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
    # width="{width}" height="{height}">
    # {''.join(svg_images)}
    # </svg>
    # """

    # # Save the SVG string to a file
    # svg_file_path = os.path.join(output_folder, "output.svg")
    # with open(svg_file_path, "w") as f:
    #     f.write(svg_str)

    # # Step 4: Create the HTML file
    # html_str = (
    #     f"<!DOCTYPE html>"
    #     f'<html lang="en">'
    #     f"<head>"
    #     f'<meta charset="UTF-8">'
    #     f'<meta name="viewport" content="width=device-width, initial-scale=1.0">'
    #     f"<title>Image Arrangement</title>"
    #     f"<style>"
    #     f"  .svg-container {{ overflow: auto}}"
    #     f"</style>"
    #     f"</head>"
    #     f"<body style=\"overflow: auto;\">"
    #     f'<div id="svg-container" style=\"width: {width}px; height: {height}px;\">'
    #     f"{svg_str}"
    #     f"</div>"
    #     f"</body>"
    #     f"</html>"
    # )

    # html_file_path = os.path.join(output_folder, "arrangement.html")
    # with open(html_file_path, "w") as f:
    #     f.write(html_str)

    # return html_file_path
