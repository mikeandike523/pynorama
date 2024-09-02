import os

import numpy as np
import scipy.linalg
from termcolor import colored

from lib.image_processing.apply_h_matrix_to_point import apply_h_matrix_to_point
from lib.image_processing.create_image_arrangement import create_image_arrangement
from lib.image_processing.RGBAImage import RGBAImage
from lib.image_processing.stitch_two import stitch_two
from lib.image_processing.warp_without_cropping import warp_without_cropping


from scipy.linalg import sqrtm
import scipy.linalg as la


def top_left_from_points(pts):
    """
    Obtains the top-eft corner of the smallest rectangle enclosing the given points
    """
    pts = np.array(pts, float)
    return np.min(pts, axis=0)


def untranslate(H, width, height):
    """
    A nonlinear transformation in Homogenous 2D coordinate space
    That will remove translation/offset from a homography transformation
    After applying to a rectangle of size with and height
    """

    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    transformed_corners = np.array(
        [apply_h_matrix_to_point(corner, H) for corner in corners]
    )
    tlc = top_left_from_points(transformed_corners)

    tmatrix = np.eye(3)
    tmatrix[:2, 2] = -tlc
    return np.dot(tmatrix, H)


def compute_arrangement(input_folder, found_files):
    """
    Performs sequential stitching of the input images
    and computes the boundaries of each image in the panorama
    """

    Hs = [np.eye(3)]

    warp_matrices = [np.eye(3)]
    tlc_seq = [np.array([0, 0], float)]

    for i, (file1, file2) in enumerate(zip(found_files[:-1], found_files[1:])):

        print(f"Processing pair {i + 1}/{len(found_files) - 1}...")

        pixels1 = RGBAImage.from_file(os.path.join(input_folder, file1), 1).pixels
        pixels2 = RGBAImage.from_file(os.path.join(input_folder, file2), 1).pixels

        height, width = pixels1.shape[:2]

        last_H = Hs[-1]
        H = stitch_two(pixels1, pixels2)

        last_tlc = tlc_seq[-1]

        delta_matrix = np.dot(H, untranslate(last_H, width, height))

        new_tlc = last_tlc + apply_h_matrix_to_point(
            np.array([0, 0], float), delta_matrix
        )

        tlc_seq.append(new_tlc)

        warp_matrix = np.dot(
            untranslate(H, width, height),
            untranslate(last_H, width, height),
        )

        warp_matrices.append(warp_matrix)

    return warp_matrices, tlc_seq


def compute_arrangement_fwd_bkwd(input_folder, found_files):

    print(colored("Starting forward pass..."))
    warp_matrices_fwd, tlc_seq_fwd = compute_arrangement(input_folder, found_files)
    print(colored("Done.", "green"))

    print(colored("Starting backward pass..."))
    warp_matrices_bkwd, tlc_seq_bkwd = compute_arrangement(
        input_folder, list(reversed(found_files))
    )
    print(colored("Done.", "green"))
    warp_matrices_bkwd = list(reversed(warp_matrices_bkwd))
    tlc_seq_bkwd = list(reversed(tlc_seq_bkwd))

    global_tlc_fwd = top_left_from_points(tlc_seq_fwd)
    global_tlc_bkwd = top_left_from_points(tlc_seq_bkwd)

    tlc_seq_fwd = np.array([tlc - global_tlc_fwd for tlc in tlc_seq_fwd])
    tlc_seq_bkwd = np.array([tlc - global_tlc_bkwd for tlc in tlc_seq_bkwd])
    mean_tlc_seq = 0.5 * (tlc_seq_fwd + tlc_seq_bkwd)
    mean_warp_matrices = [
        logarithmic_matrix_mean(fwd, bkwd)
        for fwd, bkwd in zip(warp_matrices_fwd, warp_matrices_bkwd)
    ]

    return mean_warp_matrices, mean_tlc_seq


def logarithmic_matrix_mean(A, B):
    """
    Uses logarithms and exponents to combine two matrices

    Uses functions such as np.logm and np.expm
    """
    return scipy.linalg.expm(0.5 * (scipy.linalg.logm(A) + scipy.linalg.logm(B)))


def geometric_matrix_mean(A, B):
    # Step 1: Compute the matrix product AB
    AB = np.dot(A, B)

    # Step 2: Attempt to compute the matrix square root of AB
    try:
        sqrt_AB = sqrtm(AB)

        # Step 3: Verify that the square root is valid
        # Check for NaNs
        if np.any(np.isnan(sqrt_AB)):
            raise ValueError(
                "Matrix square root contains NaN values, indicating the square root does not exist."
            )

        # Check if (sqrt_AB)^2 is close to AB
        if not np.allclose(np.dot(sqrt_AB, sqrt_AB), AB):
            raise ValueError(
                "Computed matrix square root does not satisfy the equation (sqrt_AB)^2 = AB."
            )

        # Check if the square root is real, if expected
        if np.iscomplexobj(sqrt_AB):
            raise ValueError(
                "Matrix square root is complex. Expected a real matrix square root."
            )

        # If all checks pass, return the square root
        return sqrt_AB

    except la.LinAlgError as e:
        raise ValueError(
            "Matrix square root computation failed. The square root may not exist."
        ) from e


def perform_analysis(input_folder, output_file, arrangement_downsample_factor=1):
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

    mean_warp_matrices, mean_tlc_seq = compute_arrangement_fwd_bkwd(
        input_folder, found_files
    )

    locations = list(np.array(mean_tlc_seq, float) / arrangement_downsample_factor)
    warped_images = [
        warp_without_cropping(
            RGBAImage.from_file(
                os.path.join(input_folder, found_file), arrangement_downsample_factor
            ).pixels,
            mean_warp_matrix,
        )
        for found_file, mean_warp_matrix in zip(found_files, mean_warp_matrices)
    ]

    create_image_arrangement(warped_images, locations, output_file)
