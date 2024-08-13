import copy
from types import SimpleNamespace
from typing import Protocol
import cv2
import numpy as np
from termcolor import colored

import warnings

warnings.filterwarnings("error")


from .RGBAImage import RGBAImage
from .compute_overlapping_pixels import compute_overlapping_pixels
from .apply_h_matrix_to_point import apply_h_matrix_to_point
from .warp_without_cropping import warp_without_cropping


GRADIENT_ASCENT_STEP = 2
NUM_GRADIENT_ASCENT_ITERATIONS = 10


class StitchParams(Protocol):
    BLUR_SIGMA: float
    NUM_GOOD_MATCHES: int
    GOOD_MATCH_CUTOFF: float
    NUM_TREES: int
    NUM_CHECKS: int
    RANSAC_REPROJECTION_THRESHOLD: float


STITCH_PARAMS: StitchParams = SimpleNamespace(
    BLUR_SIGMA=1.0,
    NUM_GOOD_MATCHES=32,
    GOOD_MATCH_CUTOFF=0.10,
    NUM_TREES=16,
    NUM_CHECKS=640,
    RANSAC_REPROJECTION_THRESHOLD=2.0,
)

ITERATION_TEST_STEP = 0.025


class InsufficientMatchesError(ValueError):

    def __init__(self, expected, actual):
        super().__init__(
            f"Not enough (good) SIFT matches... Expected {expected} but got {actual}"
        )
        self.expected = expected
        self.actual = actual


def stitch_two(
    A: np.ndarray,
    B: np.ndarray,
    params: StitchParams = STITCH_PARAMS,
) -> np.ndarray:
    """
    Returns the H matrix for the perspective based stitching
    of image B (sample) onto image A (reference).

    Arguments:
    A : ndarray - The reference image
    B : ndarray - The image to be transformed
    exclude_fully_transparent : boolean - Whether to exclude matches in fully transparent regions
    """

    A = (
        RGBAImage.from_pixels(A)
        .to_greyscale()
        .gaussian_blurred(params.BLUR_SIGMA)
        .pixels
    )
    B = (
        RGBAImage.from_pixels(B)
        .to_greyscale()
        .gaussian_blurred(params.BLUR_SIGMA)
        .pixels
    )

    # Create masks for fully transparent regions
    A_transparent_mask = A[:, :, 3] == 0 if A.shape[2] == 4 else None
    B_transparent_mask = B[:, :, 3] == 0 if B.shape[2] == 4 else None

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    keypoints_A, descriptors_A = sift.detectAndCompute(
        cv2.cvtColor(A, cv2.COLOR_BGRA2BGR), None
    )
    keypoints_B, descriptors_B = sift.detectAndCompute(
        cv2.cvtColor(B, cv2.COLOR_BGRA2BGR), None
    )

    keypoints_A, descriptors_A = filter_keypoints(
        keypoints_A, descriptors_A, A_transparent_mask
    )
    keypoints_B, descriptors_B = filter_keypoints(
        keypoints_B, descriptors_B, B_transparent_mask
    )

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=params.NUM_TREES)
    search_params = dict(checks=params.NUM_CHECKS)

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_B, descriptors_A, k=2)

    # Store all the good matches as per Lowe's ratio test
    good_matches = [
        m for m, n in matches if m.distance < params.GOOD_MATCH_CUTOFF * n.distance
    ]

    if len(good_matches) < params.NUM_GOOD_MATCHES:
        raise InsufficientMatchesError(params.NUM_GOOD_MATCHES, len(good_matches))

    # Extract location of good matches
    points_A = np.float32([keypoints_A[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    points_B = np.float32([keypoints_B[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )

    # Find the homography matrix
    H, _ = cv2.findHomography(
        points_B, points_A, cv2.RANSAC, params.RANSAC_REPROJECTION_THRESHOLD
    )

    return H


def compute_mse_overlap(A: np.ndarray, B: np.ndarray, H: np.ndarray) -> float:
    _, mask_B = compute_overlapping_pixels(A, B, H)

    # get a list of all nonzero coordinates (x,y) (c,r) in mask_B
    B_overlap_coords = np.flip(np.transpose(np.nonzero(mask_B)), axis=1)

    result = 0.0
    denom = np.count_nonzero(
        np.logical_and(mask_B, np.logical_and(A[:, :, 3] == 255, B[:, :, 3] == 255))
    )

    if denom == 0:
        raise ZeroDivisionError(
            "No overlapping pixels between A and transformed B by H"
        )

    num_overlap_coords = len(B_overlap_coords)

    report_interval = int(0.1 * num_overlap_coords) + 1

    for i, (x, y) in enumerate(list(B_overlap_coords)):
        Ax, Ay = np.round(apply_h_matrix_to_point(np.array([x, y], float), H))
        if Ax >= 0 and Ax < A.shape[1] and Ay >= 0 and Ay < A.shape[0]:
            AR, AG, AB, AA = A[int(Ay), int(Ax), :4]
            BR, BG, BB, BA = B[int(y), int(x), :4]
            try:
                if AA == 255 and BA == 255:
                    value = (
                        (
                            (AR / 255 - BR / 255) ** 2
                            + (AG / 255 - BG / 255) ** 2
                            + (AB / 255 - BB / 255) ** 2
                        )
                        * (255**2)
                        / 3
                    )
                    result += value
                if i % report_interval == 0:
                    print(f"Processed {i}/{num_overlap_coords} overlapping pixels")
            except RuntimeWarning as e:
                print("Arithmetic error or overflow occurred when calculating mse: {e}")
                raise e

    return result / denom


def calculate_fitness(A, B, init_H, current_corners):

    # for simplicity
    assert A.shape == B.shape, "A and B must be of the same shape"
    H, W = A.shape[:2]

    untransformed_corners = np.array([[0, 0], [W, 0], [W, H], [0, H]])

    init_mask_A, _ = compute_overlapping_pixels(A, B, init_H)

    init_overlapping_pixel_count = np.count_nonzero(init_mask_A)

    if init_overlapping_pixel_count == 0:
        raise ZeroDivisionError("No overlapping pixels between A and B with initial H")

    current_H = cv2.getPerspectiveTransform(
        np.array(current_corners).astype(np.float32),
        untransformed_corners.astype(np.float32),
    )

    current_mask_A, _ = compute_overlapping_pixels(A, B, current_H)

    current_overlapping_pixel_count = np.count_nonzero(current_mask_A)

    delta_overlapping_pixels = abs(
        current_overlapping_pixel_count - init_overlapping_pixel_count
    )

    mse = compute_mse_overlap(A, B, current_H)

    return -np.log(1 + mse + delta_overlapping_pixels / init_overlapping_pixel_count)


def filter_keypoints(keypoints, descriptors, mask):
    """
    Filter out keypoints that lie in the masked (transparent) regions.
    """
    if mask is None:
        return keypoints, descriptors

    valid_keypoints = []
    valid_descriptors = []

    for keypoint, descriptor in zip(keypoints, descriptors):
        x, y = map(int, keypoint.pt)
        if not mask[y, x]:  # If the pixel is not transparent
            valid_keypoints.append(keypoint)
            valid_descriptors.append(descriptor)
        else:
            # print(f"Skipping keypoint at ({x}, {y}) due to transparent region")
            pass

    return valid_keypoints, np.array(valid_descriptors)


def stitch_two_and_refine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Gets an initial estimate by calling stitch_two
    Then isolates the the overlapping region by changing non-overlapping pixels
    in both images to black+transparent (R=G=B=A=0)

    Then performs an iterative refinement approach to get the best match possible
    """

    init_corners = np.array(RGBAImage.from_pixels(A).corners(), float)

    tolerance_value = STITCH_PARAMS.GOOD_MATCH_CUTOFF
    while tolerance_value <= 0.5:
        print(
            f"""
Refinement step:
Attempting to find {STITCH_PARAMS.NUM_GOOD_MATCHES} matches
at {(100*tolerance_value):2.2f}% tolerance...
            """
        )
        try:
            iteration_params = copy.copy(STITCH_PARAMS)
            iteration_params.GOOD_MATCH_CUTOFF = tolerance_value
            result = stitch_two(A.copy(), B.copy(), iteration_params)
            fitness = calculate_fitness(
                A,
                B,
                result,
                np.array(
                    [
                        apply_h_matrix_to_point(point, result)
                        for point in RGBAImage.from_pixels(A).corners()
                    ],
                    float,
                ),
            )
            print(colored("Refinement successful", "green"))
            print(colored(f"fitness: {fitness}", "magenta"))
            print(colored("Performing gradient descent", "yellow"))
            init_H = result
            current_corners = np.array(
                [
                    apply_h_matrix_to_point(point, result)
                    for point in RGBAImage.from_pixels(A).corners()
                ],
                float,
            )
            for step in range(NUM_GRADIENT_ASCENT_ITERATIONS):
                print(
                    colored(
                        f"Iteration {step+1}/{NUM_GRADIENT_ASCENT_ITERATIONS}",
                        "magenta",
                    )
                )
                parameters = current_corners.reshape((-1,))
                parameter_gradient = []
                for parameter_index in range(len(parameters)):
                    print(
                        colored(
                            f"Processing parameter {parameter_index+1}/{len(parameters)}",
                            "blue",
                        )
                    )
                    new_parameter_set_minus = list(parameters.copy())
                    new_parameter_set_plus = list(parameters.copy())
                    new_parameter_set_minus[parameter_index] -= GRADIENT_ASCENT_STEP
                    new_parameter_set_plus[parameter_index] += GRADIENT_ASCENT_STEP
                    minus_fitness = calculate_fitness(
                        A,
                        B,
                        init_H,
                        np.array(new_parameter_set_minus, float).reshape((-1, 2)),
                    )
                    plus_fitness = calculate_fitness(
                        A,
                        B,
                        init_H,
                        np.array(new_parameter_set_plus, float).reshape((-1, 2)),
                    )
                    delta_fitness = plus_fitness - minus_fitness
                    parameter_gradient.append(
                        delta_fitness / (2 * GRADIENT_ASCENT_STEP)
                    )
                print(colored("Parameter Gradient", "blue"))
                for item in parameter_gradient:
                    print(colored(str(item), "blue"))
                coordinate_gradient = np.array(parameter_gradient, float).reshape(
                    (-1, 2)
                )
                grad_A, grad_B, grad_C, grad_D = coordinate_gradient
                # normalize gradients
                mag_A = np.linalg.norm(grad_A)
                if abs(mag_A) < 1e-9:
                    raise ValueError("Gradient magnitude of A is too small")
                grad_A /= mag_A
                mag_B = np.linalg.norm(grad_B)
                if abs(mag_B) < 1e-9:
                    raise ValueError("Gradient magnitude of B is too small")
                grad_B /= mag_B
                mag_C = np.linalg.norm(grad_C)
                if abs(mag_C) < 1e-9:
                    raise ValueError("Gradient magnitude of C is too small")
                grad_C /= mag_C
                mag_D = np.linalg.norm(grad_D)
                if abs(mag_D) < 1e-9:
                    raise ValueError("Gradient magnitude of D is too small")
                delta_A = grad_A * GRADIENT_ASCENT_STEP
                delta_B = grad_B * GRADIENT_ASCENT_STEP
                delta_C = grad_C * GRADIENT_ASCENT_STEP
                delta_D = grad_D * GRADIENT_ASCENT_STEP
                current_corners = np.array(
                    [
                        (np.array(corner, float) + np.array(delta, float))
                        for (corner, delta) in zip(
                            current_corners, [delta_A, delta_B, delta_C, delta_D]
                        )
                    ],
                    float,
                )
                result = cv2.getPerspectiveTransform(
                    (init_corners).astype(np.float32),
                    (current_corners).astype(np.float32),
                )
            return result
        except InsufficientMatchesError as e:
            print(colored(str(e), "red"))
        tolerance_value += ITERATION_TEST_STEP

    raise ValueError(
        "Could not find enough matches up to 50% tolerance. Stitch cannot converge."
    )
