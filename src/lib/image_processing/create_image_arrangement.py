"""
A utility to construct a large image
from a collection of sub-images and their locations
Mixes overlapping regions using the mean
"""

import base64
import os
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image

from .RGBAInfiniteMixingCanvas import RGBAInfiniteMixingCanvas


def save_image(image: np.ndarray, directory: str, filename: str) -> str:
    """Save the RGBA image as a PNG file in the specified directory."""
    img = Image.fromarray(image, "RGBA")
    file_path = os.path.join(directory, filename)
    img.save(file_path, format="PNG")
    return file_path


def create_image_arrangement(
    images: List[np.ndarray], locations: List[np.ndarray], output_file: str
) -> str:
    """
    A utility to construct a large image
    from a collection of sub-images and their locations
    Mixes overlapping regions using the mean
    """

    locations = np.array(locations, float).round().astype(int)

    tlc = np.min(locations, axis=0)

    locations = locations - tlc

    # Ensure the output directory exists, recreate if necessary
    if os.path.exists(output_file):
        os.remove(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    infinite_canvas = RGBAInfiniteMixingCanvas()

    for i, [image, location] in enumerate(zip(images, locations)):
        print(f"Processing image {i + 1}/{len(images)}...")
        infinite_canvas.put(image, *list(np.round(np.array(location)).astype(int)))

    Image.fromarray(infinite_canvas.to_RGBA()).save(output_file)


def image_to_base64(image: np.ndarray) -> str:
    """Convert a NumPy image array to a base64-encoded string."""
    img = Image.fromarray(image, "RGBA")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def create_svg_image_arrangement(
    images: List[np.ndarray], locations: List[np.ndarray], output_file: str
) -> str:
    """
    Create an SVG file containing the images positioned at the specified locations.
    Each image is embedded in the SVG as a base64-encoded string.
    """

    locations = np.array(locations, float).round().astype(int)
    tlc = np.min(locations, axis=0)
    locations = locations - tlc

    # Calculate the dimensions of the SVG based on image positions
    max_x = max(loc[0] + img.shape[1] for img, loc in zip(images, locations))
    max_y = max(loc[1] + img.shape[0] for img, loc in zip(images, locations))

    svg_content = [
        f"""
<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{max_x}" height="{max_y}">
        """
    ]

    for i, (image, location) in enumerate(zip(images, locations)):
        print(f"Processing image {i + 1}/{len(images)}...")
        img_base64 = image_to_base64(image)
        x, y = location
        svg_content.append(
            f"""
<image x="{x}" y="{y}" width="{image.shape[1]}" height="{image.shape[0]}"
href="data:image/png;base64,{img_base64}" />
"""
        )

    svg_content.append("</svg>")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_content))

    return output_file
