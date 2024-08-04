from types import SimpleNamespace
from typing import Protocol
import cv2
import numpy as np

from .apply_h_matrix_to_point import apply_h_matrix_to_point
from .RGBAImage import RGBAImage


class StitchParams(Protocol):
    NUM_GOOD_MATCHES: int
    GOOD_MATCH_CUTOFF: float
    NUM_TREES: int
    NUM_CHECKS: int
    RANSAC_REPROJECTION_THRESHOLD: float


INIT_ESTIMATE_PARAMS = SimpleNamespace(
    NUM_GOOD_MATCHES=64,
    GOOD_MATCH_CUTOFF=0.30,
    NUM_TREES=8,
    NUM_CHECKS=512,
    RANSAC_REPROJECTION_THRESHOLD=3.0,
)
REFINEMENT_PARAMS = SimpleNamespace(
    NUM_GOOD_MATCHES=64,
    GOOD_MATCH_CUTOFF=0.30,
    NUM_TREES=8,
    NUM_CHECKS=512,
    RANSAC_REPROJECTION_THRESHOLD=3.0,
)


def stitch_two(
    A: np.ndarray,
    B: np.ndarray,
    exclude_fully_transparent=True,
    params: StitchParams = INIT_ESTIMATE_PARAMS,
) -> np.ndarray:
    """
    Returns the H matrix for the perspective based stitching
    of image B (sample) onto image A (reference).

    Arguments:
    A : ndarray - The reference image
    B : ndarray - The image to be transformed
    exclude_fully_transparent : boolean - Whether to exclude matches in fully transparent regions
    """

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

    if exclude_fully_transparent:
        # Filter out keypoints in transparent regions
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
        raise ValueError(
            f"Not enough matches found - {len(good_matches)}/{params.NUM_GOOD_MATCHES}"
        )

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


def raster_closed_shape_onto_rectangle_mask(
    mask: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    """
    Rasterizes a closed shape onto a rectangle mask,
    automatically handling edge cases as well as
    shapes partially or fully outside the rectangle

    Arguments:
    mask : ndarray - The mask to rasterize the shape onto
    shape : ndarray - The closed shape to rasterize

    Returns:
    ndarray - The rasterized closed shape
    """

    # Ensure the shape is closed by converting it to an integer type
    shape = shape.astype(np.int32)

    # Create a copy of the mask to avoid modifying the original
    target = np.expand_dims(mask.copy(), -1).astype(np.uint8) * 255

    # Rasterize the shape onto the mask using OpenCV's fillPoly function
    cv2.fillPoly(target, [shape], 1)

    return target.squeeze() > 0


def stitch_two_and_refine(
    A: np.ndarray, B: np.ndarray, exclude_fully_transparent=True
) -> np.ndarray:
    """
    Gets an initial estimate by calling stitch_two
    Then isolates the the overlapping region by changing non-overlapping pixels
    in both images to black+transparent (R=G=B=A=0)
    Then calls stitch_two again with the modified images to further refine the estimate

    Note:
    The value of `exclude_fully_transparent` is applied only to the initial estimate
    The second call to stitch_two `exclude_fully_transparent` is always True
    """
    init_H = stitch_two(A.copy(), B.copy(), exclude_fully_transparent)
    init_H_inv = np.linalg.inv(init_H)
    image_A = RGBAImage.from_pixels(A.copy())
    image_B = RGBAImage.from_pixels(B.copy())
    corners_A = image_A.corners()
    corners_B = image_B.corners()
    overlap_mask_A = np.zeros(image_A.pixels.shape[:2], dtype=np.bool)
    overlap_mask_B = np.zeros(image_B.pixels.shape[:2], dtype=np.bool)
    warped_corners_A = np.array(
        [apply_h_matrix_to_point(corner, init_H_inv) for corner in corners_A], float
    )
    warped_corners_B = np.array(
        [apply_h_matrix_to_point(corner, init_H) for corner in corners_B], float
    )
    overlap_mask_A = raster_closed_shape_onto_rectangle_mask(
        overlap_mask_A, warped_corners_B
    )
    overlap_mask_B = raster_closed_shape_onto_rectangle_mask(
        overlap_mask_B, warped_corners_A
    )
    image_A.pixels[~overlap_mask_A, :] = 0
    image_B.pixels[~overlap_mask_B, :] = 0
    try:
        refined_H = stitch_two(image_A.pixels, image_B.pixels, True, REFINEMENT_PARAMS)
        return refined_H
    except ValueError as e:
        print("Failed to refine the stitching estimate.")
        print(e)
        return init_H
