import copy
from types import SimpleNamespace
from typing import Protocol
import cv2
import numpy as np
from termcolor import colored
import warnings
from tqdm import tqdm


from .RGBAImage import RGBAImage
from .compute_overlapping_pixels import compute_overlapping_pixels
from .apply_h_matrix_to_point import apply_h_matrix_to_point

# check N pixels in either direction to estimate the gradient
# this is because its empirical and not known ahead of time
# this is similar in concept to how normal vectors
# are estimated in SDF computation and raymarching
GRADIENT_ESTIMATE_RESOLUTION = 2
# Usually determined empirically
# Delta fitness is usually quite small due to logarithmic nature of fitness function
# The absolute fitness value at the initial state is generally not relevant
# An analogy may be an amplifier or gain for a sensitive instrument
STEP_PIXELS_PER_DELTA_FITNESS = 2000
# The maximum number of gradient ascent iterations
NUM_GRADIENT_ASCENT_ITERATIONS = 8
# Prevent travel of a corner if it is less (in magnitude) than this value
# If all corners dont travel, stop gradient ascent
TRAVEL_CUTOFF_PIXELS = 0.25


class StitchParams(Protocol):
    BLUR_SIGMA: float
    NUM_GOOD_MATCHES: int
    GOOD_MATCH_CUTOFF: float
    NUM_TREES: int
    NUM_CHEßCKS: int
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

    with tqdm(range(num_overlap_coords), desc="Computing MSE", unit="pixels") as pbar:

        for x, y in list(B_overlap_coords):
            Ax, Ay = np.round(apply_h_matrix_to_point(np.array([x, y], float), H))
            if Ax >= 0 and Ax < A.shape[1] and Ay >= 0 and Ay < A.shape[0]:
                AR, AG, AB, AA = A[int(Ay), int(Ax), :4]
                BR, BG, BB, BA = B[int(y), int(x), :4]
                try:
                    warnings.filterwarnings("error")

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
                    pbar.update(1)
                except RuntimeWarning as e:
                    print(
                        "Arithmetic error or overflow occurred when calculating mse: {e}"
                    )
                    raise e
                except Exception as e:
                    print("An unknown error occured while calculating mse: {e}")
                    raise e
                finally:
                    warnings.resetwarnings()
            else:
                pbar.update(1)

    return result / denom


def calculate_fitness(A, B, init_H, current_corners):

    # for simplicity
    assert A.shape == B.shape, "A and B must be of the same shape"
    H, W = A.shape[:2]

    untransformed_corners = np.array([[0, 0], [W, 0], [W, H], [0, H]],float)

    init_mask_A, _ = compute_overlapping_pixels(A, B, init_H)

    init_overlapping_pixel_count = np.count_nonzero(init_mask_A)

    if init_overlapping_pixel_count == 0:
        raise ZeroDivisionError("No overlapping pixels between A and B with initial H")

    current_H = cv2.getPerspectiveTransform(
        current_corners.astype(np.float32),
        untransformed_corners.astype(np.float32),
    )

    current_mask_A, _ = compute_overlapping_pixels(A, B, current_H)

    current_overlapping_pixel_count = np.count_nonzero(current_mask_A)

    delta_overlapping_pixels = abs(
        current_overlapping_pixel_count - init_overlapping_pixel_count
    )

    mse = compute_mse_overlap(A, B, current_H)

    # return -np.log(1 + mse + delta_overlapping_pixels / init_overlapping_pixel_count)
    return -np.log(1+mse)


def filter_keypoints(keypoints, descriptors, mask):
    """
    Filter out keypoints that lie in the masked (transparent) regions.
    """
    if mask is None:
        return keypoints, descriptors

    valid_keypoints = []
    valid_descriptors = []

    num_skipped_keypoints = 0

    for keypoint, descriptor in zip(keypoints, descriptors):
        x, y = map(int, keypoint.pt)
        if not mask[y, x]:  # If the pixel is not transparent
            valid_keypoints.append(keypoint)
            valid_descriptors.append(descriptor)
        else:
            num_skipped_keypoints += 1
            pass

    if num_skipped_keypoints > 0:
        print(f"Skipped {num_skipped_keypoints} keypoints due to transparency")

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
            init_H = stitch_two(A.copy(), B.copy(), iteration_params)

            print(
                colored(
                    f"Successfully found initial stitch estimate at tolerance {tolerance_value}",
                    "green",
                )
            )

            init_estimate_corners = np.array(
                [
                    apply_h_matrix_to_point(point, init_H)
                    for point in init_corners
                ],
                float,
            )
            current_corners = init_estimate_corners.copy()
            for step in range(NUM_GRADIENT_ASCENT_ITERATIONS):
                print(
                    colored(
                        f"Iteration {step+1}/{NUM_GRADIENT_ASCENT_ITERATIONS}",
                        "magenta",
                    )
                )
                parameters = current_corners.copy().ravel()
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
                    new_parameter_set_minus[
                        parameter_index
                    ] -= GRADIENT_ESTIMATE_RESOLUTION
                    new_parameter_set_plus[
                        parameter_index
                    ] += GRADIENT_ESTIMATE_RESOLUTION
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
                        delta_fitness / (2 * GRADIENT_ESTIMATE_RESOLUTION)
                    )
                print(colored("Parameter Gradient", "blue"))
                for item in parameter_gradient:
                    print(colored(str(item), "blue"))
                coordinate_gradient = np.array(parameter_gradient, float).copy().reshape(
                    (-1, 2)
                )
                grad_A, grad_B, grad_C, grad_D = coordinate_gradient.copy()
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
                grad_D /= mag_D
                added_fitness_A = np.linalg.norm(coordinate_gradient[0])
                added_fitness_B = np.linalg.norm(coordinate_gradient[1])
                added_fitness_C = np.linalg.norm(coordinate_gradient[2])
                added_fitness_D = np.linalg.norm(coordinate_gradient[3])
                travel_A = added_fitness_A * STEP_PIXELS_PER_DELTA_FITNESS
                travel_B = added_fitness_B * STEP_PIXELS_PER_DELTA_FITNESS
                travel_C = added_fitness_C * STEP_PIXELS_PER_DELTA_FITNESS
                travel_D = added_fitness_D * STEP_PIXELS_PER_DELTA_FITNESS
                print(colored("corner gradients (normalized/direction-only) this iteration:", "green"))
                print("x\ty")
                for grad_x,grad_y in [grad_A, grad_B, grad_C, grad_D]:
                    print(f"{grad_x:.2f}" + f"\t{grad_y:.2f}")
                print(colored("Corner travel magnitudes this iteration:", "green"))
                for travel in [travel_A, travel_B, travel_C, travel_D]:
                    print(f"{travel}"
                          +
                          f"{"(cutoff)" if abs(travel) < TRAVEL_CUTOFF_PIXELS else ""}")
                if (abs(travel_A) < TRAVEL_CUTOFF_PIXELS
                    and abs(travel_B) < TRAVEL_CUTOFF_PIXELS
                    and abs(travel_C) < TRAVEL_CUTOFF_PIXELS
                        and abs(travel_D) < TRAVEL_CUTOFF_PIXELS):
                    print(colored(f"""
Gradient ascent early stop condition reached
No corner traveled >= {TRAVEL_CUTOFF_PIXELS} pixels
""","yellow"))
                    break

                travel_A = travel_A if abs(travel_A) >= TRAVEL_CUTOFF_PIXELS else 0
                travel_B = travel_B if abs(travel_B) >= TRAVEL_CUTOFF_PIXELS else 0
                travel_C = travel_C if abs(travel_C) >= TRAVEL_CUTOFF_PIXELS else 0
                travel_A = travel_D if abs(travel_D) >= TRAVEL_CUTOFF_PIXELS else 0



                delta_A = grad_A * travel_A
                delta_B = grad_B * travel_B
                delta_C = grad_C * travel_C
                delta_D = grad_D * travel_D

                current_corners = np.array(
                    [
                        corner+delta
                        for (corner, delta) in zip(
                            list(current_corners), [delta_A, delta_B, delta_C, delta_D]
                        )
                    ],
                    float,
                )




                print(colored(f"Corner deviation after iteration {step}:", "green"))
                print("x\ty")
                for init_estimate_corner, current_corner in zip(init_estimate_corners, current_corners):
                    delta_x, delta_y = current_corner - init_estimate_corner
                    print(f"{delta_x:.2f}\t{delta_y:.2f}")

            return cv2.getPerspectiveTransform(
                (init_corners).astype(np.float32),
                (current_corners).astype(np.float32),
            )
        except InsufficientMatchesError as e:
            print(colored(str(e), "red"))
        tolerance_value += ITERATION_TEST_STEP

    raise ValueError(
        """
        Could not find initial stitch estimate (insufficient SIFT matches)
        or could not complete gradient ascent
        (e.g. init fitness too close to zero) before reaching 50% tolerance
        Generally, tolerances more than 50% are fully irrelevant
        Consider downsampling your input images
        Recommended settings are downsampling by a factor of 2
        with bilinear interpolation
        """
    )
