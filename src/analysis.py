import os
import shutil

import cv2
import numpy as np
from PIL import Image

from lib.image_processing.create_image_arrangement import create_image_arrangement
from lib.image_processing.warp_without_cropping import warp_without_cropping
from lib.image_processing.apply_h_matrix_to_point import apply_h_matrix_to_point
from lib.image_processing.RGBAImage import RGBAImage
from lib.image_processing.stitch_two import stitch_two
from lib.image_processing.RGBAInfiniteMixingCanvas import RGBAInfiniteMixingCanvas


def top_left_from_points(pts):
    """
    Obtains the top-eft corner of the smallest rectangle enclosing the given points
    """
    pts = np.array(pts, float)
    return np.min(pts, axis=0)


def compute_arrangement(input_folder, found_files):
    """
    Performs sequential stitching of the input images
    and computes the boundaries of each image in the panorama
    """

    debug_pairs_folder = os.path.join(os.getcwd(), "testing", "output", "debugpairs")

    if os.path.exists(debug_pairs_folder):
        shutil.rmtree(debug_pairs_folder)

    os.makedirs(debug_pairs_folder, exist_ok=True)

    Hs = []

    init_image = RGBAImage.from_file(os.path.join(input_folder, found_files[0]), 1)

    init_A, init_B, init_C, init_D = init_image.corners()

    corner_seq_A = [init_A]
    corner_seq_B = [init_B]
    corner_seq_C = [init_C]
    corner_seq_D = [init_D]

    for i, (file1, file2) in enumerate(zip(found_files[:-1], found_files[1:])):

        print(f"Processing pair {i + 1}/{len(found_files) - 1}...")

        last_H = Hs[-1] if Hs else np.eye(3)

        pixels1 = RGBAImage.from_file(os.path.join(input_folder, file1), 1).pixels
        pixels2 = RGBAImage.from_file(os.path.join(input_folder, file2), 1).pixels

        pixels1 = warp_without_cropping(pixels1.copy(), last_H)
        pixels2 = warp_without_cropping(pixels2.copy(), last_H)

        H = stitch_two(pixels1, pixels2)
        debug_canvas = RGBAInfiniteMixingCanvas()

        dpx1 = pixels1.copy()
        dpx2 = warp_without_cropping(pixels2.copy(), H)
        dpx2tlc = top_left_from_points(
            np.array(
                [
                    apply_h_matrix_to_point(corner, H)
                    for corner in RGBAImage.from_pixels(pixels2).corners()
                ]
            )
        )

        debug_canvas.put(dpx1, 0, 0)
        debug_canvas.put(dpx2, int(round(dpx2tlc[0])), int(round(dpx2tlc[1])))

        debug_name = f"debug_pair_{i + 1}.png"

        Image.fromarray(debug_canvas.to_RGBA()).save(
            os.path.join(debug_pairs_folder, debug_name)
        )

        Hs.append(H)

        last_corner_A = corner_seq_A[-1].copy()
        last_corner_B = corner_seq_B[-1].copy()
        last_corner_C = corner_seq_C[-1].copy()
        last_corner_D = corner_seq_D[-1].copy()

        last_tlc = top_left_from_points(
            np.array([last_corner_A, last_corner_B, last_corner_C, last_corner_D])
        )

        corner_deltas = np.array(
            [
                apply_h_matrix_to_point(corner, H)
                for corner in RGBAImage.from_pixels(pixels2).corners()
            ]
        )
        delta_A, delta_B, delta_C, delta_D = list(corner_deltas)

        next_A = last_tlc + delta_A
        next_B = last_tlc + delta_B
        next_C = last_tlc + delta_C
        next_D = last_tlc + delta_D

        corner_seq_A.append(next_A)
        corner_seq_B.append(next_B)
        corner_seq_C.append(next_C)
        corner_seq_D.append(next_D)

        print(next_A, next_B, next_C, next_D)

    print("Obtaining panorama segment boundaries...")

    boundaries = [
        np.array(corners, dtype=float)
        for corners in zip(corner_seq_A, corner_seq_B, corner_seq_C, corner_seq_D)
    ]

    all_boundary_points = []

    for boundary in boundaries:
        all_boundary_points.extend(list(boundary))

    tlc = top_left_from_points(all_boundary_points)

    boundaries = [boundaries - tlc for boundaries in boundaries]

    return boundaries


def perform_analysis(input_folder, output_file):
    """
    Collects the input files
    computes the boundaries using sequential stitching
    arranges the images into a panorama and saves the result
    """

    if os.path.exists(output_file):
        if not os.path.isfile(output_file):
            raise ValueError(f"{output_file} already exists and is not a file.")
        os.remove(output_file)

    permitted_extensions = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".gif", ".bmp"]

    found_files = [
        file
        for file in os.listdir(input_folder)
        if any(file.lower().endswith(ext) for ext in permitted_extensions)
    ]

    found_integers = []
    found_integers_files = {}

    for file in found_files:
        without_ext = os.path.splitext(file)[0]
        without_ext = without_ext.lstrip("0")
        if without_ext == "":
            without_ext = "0"
        try:
            number = int(without_ext)
            found_integers.append(number)
            found_integers_files[number] = file
        except ValueError as e:
            raise ValueError(
                f"""File {file} was not an integer.
Please put the files in order with names that contain only integers.
"""
            ) from e

    if len(found_integers) < 2:
        raise ValueError("At least two valid files are needed.")

    found_integers.sort()

    lowest = found_integers[0]
    highest = found_integers[-1]

    missing = []

    for i in range(lowest, highest + 1):
        if i not in found_integers:
            missing.append(i)

    if len(missing) > 0:
        raise ValueError(
            f"""
Missing one or files in the sequence.

The lowest image index in your folder is {lowest}.
The highest image index in your folder is {highest}.

You are missing the following files:

{", ".join(map(str, missing))}
"""
        )

    print(f"Found files for image index range {lowest} to {highest} (inclusive):")
    found_files = [found_integers_files[i] for i in found_integers]
    for file in found_files:
        print(file)

    init_image = RGBAImage.from_file(os.path.join(input_folder, found_files[0]), 1)

    boundaries = compute_arrangement(input_folder, found_files)

    anchors = [top_left_from_points(boundary) for boundary in boundaries]

    deltas = [boundary - anchor for boundary, anchor in zip(boundaries, anchors)]

    warped_images = []
    locations = []

    for anchor, delta, found_file in zip(anchors, deltas, found_files):

        src = np.array(init_image.corners(), float).astype(np.float32)
        dst = (delta).astype(np.float32)

        H = cv2.getPerspectiveTransform(src, dst)

        image = RGBAImage.from_file(os.path.join(input_folder, found_file), 1)

        warped_image = warp_without_cropping(image.pixels, H)
        warped_images.append(warped_image)
        locations.append(anchor)

    create_image_arrangement(warped_images, locations, output_file)
