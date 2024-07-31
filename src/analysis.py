import os
from functools import reduce

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


def top_left_from_points(pts):
    pts = np.array(pts, float)
    return np.min(pts, axis=0)


def points_range_x(pts):
    return np.max(pts[:, 0]) - np.min(pts[:, 0])


def points_range_y(pts):
    return np.max(pts[:, 1]) - np.min(pts[:, 1])


def top_right_from_points(pts):
    pts = np.array(pts, float)
    tl = top_left_from_points(pts)
    return tl + np.array([points_range_x(pts), 0])


def bottom_left_from_points(pts):
    pts = np.array(pts, float)
    tl = top_left_from_points(pts)
    return tl + np.array([0, points_range_y(pts)])


def bottom_right_from_points(pts):
    pts = np.array(pts, float)
    tl = top_left_from_points(pts)
    return tl + np.array([points_range_x(pts), points_range_y(pts)])


def extent_corners_from_points(pts):
    return np.array(
        [
            top_left_from_points(pts),
            top_right_from_points(pts),
            bottom_right_from_points(pts),
            bottom_left_from_points(pts),
        ],
        float,
    )


def perform_analysis_pass(input_folder, found_files):
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

        tlc = top_left_from_points(
            [last_corner_A, last_corner_B, last_corner_C, last_corner_D]
        )

        delta_next_corner_A = apply_h_matrix_to_point(init_A.copy(), H)
        delta_next_corner_B = apply_h_matrix_to_point(init_B.copy(), H)
        delta_next_corner_C = apply_h_matrix_to_point(init_C.copy(), H)
        delta_next_corner_D = apply_h_matrix_to_point(init_D.copy(), H)

        next_corner_A = tlc + delta_next_corner_A
        next_corner_B = tlc + delta_next_corner_B
        next_corner_C = tlc + delta_next_corner_C
        next_corner_D = tlc + delta_next_corner_D

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

    return boundary_sequence, H_seq


def chain_transformation_sequence(transformations):
    """
    Does a chain of left-multiples so the first item in the sequence is applied first
    """
    if len(transformations) == 0:
        return np.eye(3, 3)
    if len(transformations) == 1:
        return transformations[0]
    value = transformations[0]
    for transformation in transformations[1:]:
        value = np.dot(transformation, value)
    return value


def perform_analysis(input_folder, output_file):

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

    warped_images = []
    locations = []

    # boundary_sequence, H_seq = perform_analysis_pass(input_folder, found_files)

    # for boundary, found_file in zip(boundary_sequence, found_files):
    #     image = RGBAImage.from_file(os.path.join(input_folder, found_file))
    #     src = np.array(init_image.corners()).astype(np.float32)
    #     dst = boundary.copy()
    #     dst = (dst - top_left_from_points(boundary)).astype(np.float32)
    #     warp_H = cv2.getPerspectiveTransform(src, dst)
    #     warped_image = warp_without_cropping(image.pixels, warp_H)
    #     warped_images.append(warped_image)
    #     locations.append(top_left_from_points(boundary).round().astype(int))

    H_seq = [np.eye(3, 3).astype(float)]

    for i in range(len(found_files) - 1):
        ia = i
        ib = i + 1
        image_a_path = os.path.join(input_folder, found_files[ia])
        image_b_path = os.path.join(input_folder, found_files[ib])
        image_a = RGBAImage.from_file(image_a_path)
        image_b = RGBAImage.from_file(image_b_path)
        transformation = stitch_two(image_a.pixels, image_b.pixels)
        print(i, len(found_files), transformation)
        last_H = H_seq[-1]
        H_seq.append(np.dot(transformation, last_H))

    print([item.shape for item in H_seq])

    for found_file, H in zip(found_files, H_seq):
        image = RGBAImage.from_file(os.path.join(input_folder, found_file))
        warped_image = warp_without_cropping(image.pixels, H)
        print(found_file, warped_image.shape)
        warped_images.append(warped_image)
        location = apply_h_matrix_to_point(image.corners()[0].copy(), H)
        print(found_file, location)
        locations.append(location)

    tlc = top_left_from_points(np.array(locations, float))

    locations = [location - tlc for location in locations]

    create_image_arrangement(warped_images, locations, output_file)
