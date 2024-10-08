import os
import shutil

import cv2
import numpy as np
from termcolor import colored
from PIL import Image

from lib.image_processing import (
    PixelDownsampler,
    RGBAImage,
    stitch_two_and_refine,
    apply_h_matrix_to_point,
    warp_without_cropping,
)
from lib import make_fresh_folder
from lib.image_processing.debugging import boundary_svg
from lib.image_processing import (
    create_image_arrangement,
    PixelDownsampler,
    SupportedResamplingAlgorithm,
    RGBAInfiniteMixingCanvas,
)
from lib.filesystem import get_random_unique_in_folder


def weighted_mean_of_numpy_arrays(arrays, weights, epsilon=1e-9):
    if len(arrays) != len(weights):
        raise ValueError("Number of arrays and weights must match.")
    if len(arrays) == 0:
        raise ValueError("No arrays provided.")
    if len(arrays) == 1:
        raise ValueError(
            "Only one array provided. Behavior in this case is not clearly defined."
        )
    if np.any(~np.isfinite(weights)) or np.any(np.array(weights) < 0):
        raise ValueError("Weights must be finite and non-negative.")
    if np.sum(weights) == 0:
        raise ValueError("At least one non-zero weight required.")
    if np.abs(np.sum(weights)) < epsilon:
        raise ValueError("Weights sum is very close to zero.")
    ndim = arrays[0].ndim
    if any([array.ndim != ndim for array in arrays]):
        raise ValueError("All arrays must have the same number of dimensions.")
    shape = arrays[0].shape
    if any([array.shape != shape for array in arrays]):
        raise ValueError("All arrays must have the same shape.")
    total = weights[0] * arrays[0].copy()
    for array, weight in zip(arrays[1:], weights[1:]):
        total += weight * array
    total /= np.sum(weights)
    return total


def selection_of_numpy_arrays(arrays, weights):
    if len(arrays) != len(weights):
        raise ValueError("Number of arrays and weights must match.")
    if len(arrays) == 0:
        raise ValueError("No arrays provided.")
    if len(arrays) == 1:
        return arrays[0]
    if np.any(~np.isfinite(weights)) or np.any(np.array(weights) < 0):
        raise ValueError("Weights must be finite and non-negative.")
    ndim = arrays[0].ndim
    if any([array.ndim != ndim for array in arrays]):
        raise ValueError("All arrays must have the same number of dimensions.")
    shape = arrays[0].shape
    if any([array.shape != shape for array in arrays]):
        raise ValueError("All arrays must have the same shape.")
    index_of_max_weight = np.argmax(weights)
    return arrays[index_of_max_weight]


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


def center_from_points(pts):
    pts = np.array(pts, float)
    return np.mean(pts, axis=0)


def perform_analysis_pass(input_folder, found_files):
    print("Testing pair stitchability...")

    Hs = []

    init_image = RGBAImage.from_file(os.path.join(input_folder, found_files[0]))

    init_W = init_image.get_width()
    init_H = init_image.get_height()


    for i, (file1, file2) in enumerate(zip(found_files[:-1], found_files[1:])):
        pixels1 = RGBAImage.from_file(os.path.join(input_folder, file1)).pixels
        pixels2 = RGBAImage.from_file(os.path.join(input_folder, file2)).pixels

        dpx1 = pixels1.copy()
        dpx2 = pixels2.copy()

        H = stitch_two_and_refine(pixels1, pixels2)

        debug_folder = "testing/output/debugpairs"

        debug_fn = os.path.join(
            debug_folder, os.path.basename(input_folder) + "_" + f"pair{i}-{i+1}.png"
        )

        dcanvas = RGBAInfiniteMixingCanvas()

        debugx1 = 0
        debugy1 = 0
        debugx2, debugy2 = apply_h_matrix_to_point(
            np.array([debugx1, debugy1], float), H
        )
        dp1 = np.array([debugx1, debugy1], int)
        dp2 = np.array([debugx2, debugy2], int)
        dcanvas.put(dpx1, dp1[0], dp1[1])

        dcanvas.put(warp_without_cropping(dpx2, H), dp2[0], dp2[1])

        Image.fromarray(dcanvas.to_RGBA()).save(debug_fn)

        Hs.append(H)

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

    for H in Hs:

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

    if os.path.exists(output_file):
        if not os.path.isfile(output_file):
            raise ValueError(f"{output_file} already exists and is not a file.")
        os.remove(output_file)

    if os.path.exists("testing/output/debugpairs"):
        shutil.rmtree("testing/output/debugpairs")
    os.makedirs("testing/output/debugpairs")

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

    init_image = RGBAImage.from_file(os.path.join(input_folder, found_files[0]))

    boundaries = perform_analysis_pass(input_folder, found_files)

    anchors = [np.mean(boundary, axis=0) for boundary in boundaries]

    deltas = [boundary - anchor for boundary, anchor in zip(boundaries, anchors)]

    warped_images = []
    locations = []

    for anchor, delta, found_file in zip(anchors, deltas, found_files):
        tlc = anchor + delta[0]

        src = np.array(init_image.corners(), float).astype(np.float32)
        dst = (delta - tlc).astype(np.float32)

        H = cv2.getPerspectiveTransform(src, dst)

        image = RGBAImage.from_file(os.path.join(input_folder, found_file))
        warped_image = warp_without_cropping(image.pixels, H)
        warped_images.append(warped_image)
        locations.append(tlc)

    create_image_arrangement(warped_images, locations, output_file)
