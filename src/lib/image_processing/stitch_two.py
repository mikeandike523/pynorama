import copy
from types import SimpleNamespace
from typing import Protocol
import cv2
import numpy as np
from termcolor import colored

from .RGBAImage import RGBAImage


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


def stitch_two_and_refine(
    A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """
    Gets an initial estimate by calling stitch_two
    Then isolates the the overlapping region by changing non-overlapping pixels
    in both images to black+transparent (R=G=B=A=0)

    Then performs an iterative refinement approach to get the best match possible
    """

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
            print(colored("Refinement successful", "green"))
            return result
        except InsufficientMatchesError as e:
            print(colored(str(e), "red"))
        tolerance_value += ITERATION_TEST_STEP

    raise ValueError(
        "Could not find enough matches up to 50% tolerance. Stitch cannot converge."
    )
