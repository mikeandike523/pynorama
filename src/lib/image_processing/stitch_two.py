import copy
from types import SimpleNamespace
from typing import Optional, Protocol
import cv2
import numpy as np
from termcolor import colored
import warnings
from tqdm import tqdm


from .RGBAImage import RGBAImage
from .compute_overlapping_pixels import compute_overlapping_pixels
from .apply_h_matrix_to_point import apply_h_matrix_to_point
from .warp_without_cropping import warp_without_cropping

# check N pixels in either direction to estimate the gradient
# this is because its empirical and not known ahead of time
# this is similar in concept to how normal vectors
# are estimated in SDF computation and raymarching
GRADIENT_ESTIMATE_RESOLUTION = 2
# Usually determined empirically
# Delta fitness is usually quite small due to logarithmic nature of fitness function
# The absolute fitness value at the initial state is generally not relevant
# An analogy may be an amplifier or gain for a sensitive instrument
GAIN = 1000
# The maximum number of gradient ascent iterations
NUM_GRADIENT_ASCENT_ITERATIONS = 20
# Prevent travel of a corner if it is less (in magnitude) than this value
# If all corners dont travel, stop gradient ascent
TRAVEL_CUTOFF_PIXELS = 0.05


class StitchParams(Protocol):
    BLUR_SIGMA: float
    NUM_GOOD_MATCHES: int
    GOOD_MATCH_CUTOFF: float
    NUM_TREES: int
    NUM_CHECKS: int
    RANSAC_REPROJECTION_THRESHOLD: float


STITCH_PARAMS: StitchParams = SimpleNamespace(
    BLUR_SIGMA=2.0,
    NUM_GOOD_MATCHES=32,
    GOOD_MATCH_CUTOFF=0.10,
    NUM_TREES=8,
    NUM_CHECKS=512,
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
    """
    Computes the MSE of the overlappoign region between A and B warped by H

    for the time being, A and B must be the same shape
    However this can be improved in future versions

    Arguments:
    A: A reference image, RGBA pixels (H, W, 4) uint8
    B: A target image, RGBA pixels (H, W, 4) uint8
    H: Describe how to warp B (target) onto the space established by A (reference)
       Commonly used in image stitching, where H is a 3x3 matrix describing how
       to transform and reposition B into its best position on A to stitch the images together
    """

    #for simplicity
    assert A.shape == B.shape, "A and B must have the same shape"

    # Already implemented and tested elsewhere
    # produces "complemetary" masks for A onto B and B onto A
    mask_A, mask_B = compute_overlapping_pixels(A, B, H)

    A = A.copy()
    B = B.copy()
    A[mask_A == 0,:] = 0
    B[mask_B == 0,:] = 0

    # Already implemented and tested elsewhere
    # Is a one-liner to do matrix multiplication and W division in one step
    B_points=np.array([
        apply_h_matrix_to_point(corner, H) for corner in RGBAImage.from_pixels(B).corners()
    ],float)
    B_tlc=np.min(B_points, axis=0)
    warped_B = warp_without_cropping(B, H)
    warped_B_H, warped_B_W = warped_B.shape[:2]

    B_tlc_x, B_tlc_y = int(B_tlc[0]), int(B_tlc[1])
    
    start_x = min(max(B_tlc_x, 0),A.shape[1])
    start_y = min(max(B_tlc_y, 0),A.shape[0])
    init_end_x = B_tlc_x + warped_B_W
    init_end_y = B_tlc_y + warped_B_H
    end_x = min(max(init_end_x, 0), A.shape[1])
    end_y = min(max(init_end_y, 0), A.shape[0])

    if start_x - end_x == 0:
        raise ZeroDivisionError("Warped B does not overlap A at all along the x-axis")
    
    if start_y - end_y == 0:
        raise ZeroDivisionError("Warped B does not overlap A at all along the y-axis")

    reference_subimage = A[start_y:end_y,start_x:end_x ,:]

    subimage_H, subimage_W = reference_subimage.shape[:2]

    warped_B_subimage = warped_B[0:subimage_H,  0:subimage_W, :]

    submask= mask_A[start_y:end_y,start_x:end_x]

    denom= np.count_nonzero(submask)

    if denom == 0:
        raise ZeroDivisionError(
            "After transforming by H, B does not overlap with A"
        )
        
    subimage_A_R, subimage_A_G, subimage_A_B = (reference_subimage[:,:,c] for c in range(3))
    subimage_warped_B_R, subimage_warped_B_G, subimage_warped_B_B = (warped_B_subimage[:,:,c] for c in range(3))

    R1, G1, B1 = subimage_A_R, subimage_A_G, subimage_A_B
    R2, G2, B2 = subimage_warped_B_R, subimage_warped_B_G, subimage_warped_B_B

    R1, G1, B1 = (channel.astype(float)/255 for channel in (R1, G1, B1))
    R2, G2, B2 = (channel.astype(float)/255 for channel in (R2, G2, B2))

    total_square_error_R = np.sum(np.power(R1-R2,2))
    total_square_error_G = np.sum(np.power(G1-G2,2))
    total_square_error_B = np.sum(np.power(B1-B2,2))

    mse = (total_square_error_R + total_square_error_G + total_square_error_B) / (3* denom)

    return mse

def calculate_fitness(A, B, init_H, current_corners):

    # for simplicity
    assert A.shape == B.shape, "A and B must be of the same shape"
    H, W = A.shape[:2]

    untransformed_corners = np.array([[0, 0], [W, 0], [W, H], [0, H]],float)

    current_H = cv2.getPerspectiveTransform(
        current_corners.astype(np.float32),
        untransformed_corners.astype(np.float32),
    )

    def calc_num_overlapping_pixels_at_H(H):
        mask_A, _ = compute_overlapping_pixels(A, B, H)
        return np.count_nonzero(mask_A)
    
    init_overlapping_pixels = calc_num_overlapping_pixels_at_H(init_H)
    current_overlapping_pixels = calc_num_overlapping_pixels_at_H(current_H)

    if init_overlapping_pixels == 0:
        raise ZeroDivisionError("No overlapping pixels found at initial H")

    scaled_delta_overlapping_pixels = abs(current_overlapping_pixels - init_overlapping_pixels)/init_overlapping_pixels
 

    try:

        mse = compute_mse_overlap(RGBAImage.from_pixels(A).to_greyscale().pixels, 
                                  RGBAImage.from_pixels(B).to_greyscale().pixels
                                  , current_H)


        print(f"MSE: {mse}")
        print(f"Scaled delta overlapping pixels: {scaled_delta_overlapping_pixels}")



        fitness = -np.log(1+mse+scaled_delta_overlapping_pixels)


        print(f"Fitness: {fitness}")

        return fitness
    
    
    except ZeroDivisionError as e:
        print(f"Error while calculating fitness: {e}")
        return None


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

            return init_H

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
                        init_H.copy(),
                        np.array(new_parameter_set_minus, float).reshape((-1, 2)),
                    )
                    plus_fitness = calculate_fitness(
                        A,
                        B,
                        init_H.copy(),
                        np.array(new_parameter_set_plus, float).reshape((-1, 2)),
                    )
                    minus_fitness = minus_fitness if minus_fitness is not None else 0
                    plus_fitness = plus_fitness if plus_fitness is not None else 0
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

                def get_normalized_or_zero(grad_vector):
                    mag = np.linalg.norm(grad_vector)
                    if mag < 1e-9:
                        return np.zeros_like(grad_vector)
                    else:
                        return grad_vector / mag

                grad_A, grad_B, grad_C, grad_D = coordinate_gradient.copy()

                grad_A, grad_B, grad_C, grad_D = (
                    get_normalized_or_zero(grad_A),
                    get_normalized_or_zero(grad_B),
                    get_normalized_or_zero(grad_C),
                    get_normalized_or_zero(grad_D)
                )
                
                added_fitness_A = np.linalg.norm(coordinate_gradient[0])
                added_fitness_B = np.linalg.norm(coordinate_gradient[1])
                added_fitness_C = np.linalg.norm(coordinate_gradient[2])
                added_fitness_D = np.linalg.norm(coordinate_gradient[3])
                travel_A = added_fitness_A * GAIN
                travel_B = added_fitness_B * GAIN
                travel_C = added_fitness_C * GAIN
                travel_D = added_fitness_D * GAIN
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




                print(colored(f"Corner deviation after iteration {step+1}:", "green"))
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
