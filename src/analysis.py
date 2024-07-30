import os

import cv2
import numpy as np

from lib.image_processing import (
    RGBAImage,
    RGBADownsampler,
    stitch_two,
    apply_h_matrix_to_point,
    warp_without_cropping,
)
from lib import make_fresh_folder
from lib.image_processing.debugging import boundary_svg
from lib.image_processing import create_image_arrangement


def perform_analysis(input_folder, output_folder):

    if not os.path.exists(input_folder):
        raise FileNotFoundError("Input folder path does not exist.")

    if not os.path.isdir(input_folder):
        raise ValueError("Input folder path is not a directory.")

    make_fresh_folder(output_folder)

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
Missing one or more files:
                         
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

    print("Testing pair stitchability...")

    H_seq = []

    for file1, file2 in zip(found_files[:-1], found_files[1:]):
        pixels1 = RGBAImage.from_file(os.path.join(input_folder, file1)).pixels
        pixels2 = RGBAImage.from_file(os.path.join(input_folder, file2)).pixels

        if len(H_seq) > 0:
            last_H = H_seq[-1]
            pixels1 = warp_without_cropping(pixels1, np.linalg.inv(last_H))

        H = stitch_two(pixels1, pixels2)

        H_seq.append(H)

        print(H)

    print("Obtaining panorama segment boundaries...")

    # A = topleft corner
    # B = topright corner
    # C = bottomright corner
    # D = bottomleft corner

    init_image = RGBAImage.from_file(os.path.join(input_folder, found_files[0]))

    init_A, init_B, init_C, init_D = init_image.corners()

    corner_seq_A = [init_A]
    corner_seq_B = [init_B]
    corner_seq_C = [init_C]
    corner_seq_D = [init_D]

    for H in H_seq:

        last_corner_A = corner_seq_A[-1]
        last_corner_B = corner_seq_B[-1]
        last_corner_C = corner_seq_C[-1]
        last_corner_D = corner_seq_D[-1]

        delta_last_corner_A = last_corner_A - last_corner_A
        delta_last_corner_B = last_corner_B - last_corner_A
        delta_last_corner_C = last_corner_C - last_corner_A
        delta_last_corner_D = last_corner_D - last_corner_A

        delta_next_corner_A = apply_h_matrix_to_point(delta_last_corner_A, H)
        delta_next_corner_B = apply_h_matrix_to_point(delta_last_corner_B, H)
        delta_next_corner_C = apply_h_matrix_to_point(delta_last_corner_C, H)
        delta_next_corner_D = apply_h_matrix_to_point(delta_last_corner_D, H)

        next_corner_A = last_corner_A + delta_next_corner_A
        next_corner_B = last_corner_A + delta_next_corner_B
        next_corner_C = last_corner_A + delta_next_corner_C
        next_corner_D = last_corner_A + delta_next_corner_D

        corner_seq_A.append(next_corner_A)
        corner_seq_B.append(next_corner_B)
        corner_seq_C.append(next_corner_C)
        corner_seq_D.append(next_corner_D)

    print(corner_seq_A)
    print(corner_seq_B)
    print(corner_seq_C)
    print(corner_seq_D)

    boundary_sequence = [
        np.array(corners, dtype=float)
        for corners in zip(corner_seq_A, corner_seq_B, corner_seq_C, corner_seq_D)
    ]

    min_x = np.inf
    min_y = np.inf

    for boundary in boundary_sequence:
        min_x = min(min_x, np.min(boundary[:, 0]))
        min_y = min(min_y, np.min(boundary[:, 1]))

    print("Image boundary at the top left corner:", (min_x, min_y))

    boundary_sequence = [
        boundary
        - np.array([(min_x, min_y) for _ in range(boundary.shape[0])], dtype=float)
        for boundary in boundary_sequence
    ]

    svg_text = boundary_svg(boundary_sequence)

    with open(os.path.join(output_folder, "boundary.svg"), "w") as fl:
        fl.write(svg_text)

    boundary_sequence_xsmall = [
        1 / 16 * boundary.copy() for boundary in boundary_sequence
    ]

    svg_text = boundary_svg(boundary_sequence_xsmall)

    with open(os.path.join(output_folder, "boundary_xsmall.svg"), "w") as fl:
        fl.write(svg_text)

    images = []

    for found_file in found_files:
        images.append(RGBAImage.from_file(os.path.join(input_folder, found_file)))

    warped_images = []
    locations = []

    for boundary, image, H in zip(boundary_sequence, images, H_seq):
        A, B, C, D = boundary
        dA = A - A
        dB = B - A
        dC = C - A
        dD = D - A
        # warp_H = cv2.getPerspectiveTransform(
        #     np.array([dA, dB, dC, dD], np.float32),
        #     np.array(image.corners(), np.float32),
        # )
        warp_H = np.linalg.inv(H)
        warped_image = warp_without_cropping(image.pixels, warp_H)
        warped_images.append(warped_image)
        locations.append(A)

    create_image_arrangement(
        warped_images, locations, os.path.join(output_folder, "arrangement.png")
    )
