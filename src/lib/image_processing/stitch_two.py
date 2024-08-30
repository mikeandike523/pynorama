import copy
from types import SimpleNamespace
from typing import Protocol
import cv2
import numpy as np
from termcolor import colored


from .RGBAImage import RGBAImage
from .compute_mse_overlap import compute_mse_overlap


class StitchParams(Protocol):
    """
    An interface descirbing the relevant parameters
    for SIFT + RANSAC based perspective statiching
    """

    BLUR_SIGMA: float
    NUM_GOOD_MATCHES: int
    GOOD_MATCH_CUTOFF: float
    NUM_TREES: int
    NUM_CHECKS: int
    RANSAC_REPROJECTION_THRESHOLD: float


STITCH_PARAMS: StitchParams = SimpleNamespace(
    BLUR_SIGMA=1.0,
    NUM_GOOD_MATCHES=18,
    GOOD_MATCH_CUTOFF=0.10,
    NUM_TREES=6,
    NUM_CHECKS=480,
    RANSAC_REPROJECTION_THRESHOLD=3.0,
)

ITERATION_TEST_STEP = 0.050
CUTOFF_TOLERANCE = 0.5


class InsufficientMatchesError(ValueError):
    """
    An error thrown when not enough good SIFT matches are found
    with the given stitch parameters
    """

    def __init__(self, expected, actual):
        super().__init__(
            f"Not enough (good) SIFT matches... Expected {expected} but got {actual}"
        )
        self.expected = expected
        self.actual = actual


def precompute_features(A: np.ndarray, B: np.ndarray, params):

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

    return matches, keypoints_A, descriptors_A, keypoints_B, descriptors_B


def attempt_stitch_two_with_params(
    A: np.ndarray,
    B: np.ndarray,
    params: StitchParams,
    matches: any,
    keypoints_A: any,
    descriptors_A: any,
    keypoints_B: any,
    descriptors_B: any,
) -> np.ndarray:
    """
    Returns the H matrix for the perspective based stitching
    of image B (sample) onto image A (reference).

    Arguments:
    A : ndarray - The reference image
    B : ndarray - The image to be transformed
    """



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

    if num_skipped_keypoints > 0:
        print(f"Skipped {num_skipped_keypoints} keypoints due to transparency")

    return valid_keypoints, np.array(valid_descriptors)


def stitch_two(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calls stitch_two in a loop with progressively increasing tolerances until
    enough matches are found, then returns the resulting H (B target onto A reference)

    """

    tolerance_value = STITCH_PARAMS.GOOD_MATCH_CUTOFF
    num_matches = STITCH_PARAMS.NUM_GOOD_MATCHES

    matches, keypoints_A, descriptors_A, keypoints_B, descriptors_B = precompute_features(
        A, B, STITCH_PARAMS
    )

    while tolerance_value < 1.0 and num_matches >= 4:
        print(
            colored(
                f"Attempting to find stitch matrix as {100 * tolerance_value:.2f}% and {num_matches} matches found",
                "blue",
            )
        )
        try:
            iteration_params = copy.copy(STITCH_PARAMS)
            iteration_params.GOOD_MATCH_CUTOFF = tolerance_value
            iteration_params.NUM_GOOD_MATCHES = num_matches
            result = attempt_stitch_two_with_params(
                A.copy(),
                B.copy(),
                iteration_params,
                matches,
                keypoints_A,
                descriptors_A,
                keypoints_B,
                descriptors_B,
            )

            print(colored("Found stitch matrix", "green"))

            return result

        except InsufficientMatchesError as e:
            print(colored(str(e), "red"))
        tolerance_value += ITERATION_TEST_STEP
        if tolerance_value > CUTOFF_TOLERANCE:
            tolerance_value = STITCH_PARAMS.GOOD_MATCH_CUTOFF
            num_matches //= 2

    raise ValueError(
        "Could not find initial stitch estimate before reaching 100% tolerance or less than 4 atches"
    )
