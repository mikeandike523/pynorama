"""
A utility to construct a large image
from a collection of sub-images and their locations
Mixes overlapping regions using the mean
"""

import os
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
