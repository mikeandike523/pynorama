"""
A utility for obtaining masks that represent
the overlapping region of two images
after the second image is transformed using a homography matrix
"""

from typing import Tuple

import numpy as np
from rasterio import features
from shapely.geometry import Polygon

from .apply_h_matrix_to_point import apply_h_matrix_to_point


def create_polygon(corners: np.ndarray) -> Polygon:
    """Create a Shapely polygon from given corners."""
    return Polygon(corners)


def transform_corners(corners: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Transform corners using the given homography matrix."""
    return np.array(
        [apply_h_matrix_to_point(point, homography) for point in corners], float
    )


def compute_intersection(polygon1: Polygon, polygon2: Polygon) -> Polygon:
    """Compute the intersection between two polygons."""
    return polygon1.intersection(polygon2)


def rasterize_polygon(polygon: Polygon, shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon into a binary mask."""
    return (
        features.rasterize([(polygon, 1)], out_shape=shape, dtype=np.uint8).squeeze()
        > 0
    )


def compute_overlapping_pixels(
    image_a: np.ndarray, image_b: np.ndarray, homography: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute overlapping pixels between two images using a homography matrix.

    Arguments:
    A: pixels for an RGBA image (H,W,4)
    B: pixels for an RGBA image (H,W,4)
    H: Homography matrix (3,3)

    Returns:
    (Tuple)
    mask_A: mask for overlapping region in A
    mask_B: mask for overlapping region in B
    """
    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]

    b_target_onto_a_reference_h = homography
    b_target_onto_a_reference_h = np.linalg.inv(homography)

    init_corners_a = np.array(
        [[0, 0], [width_a, 0], [width_a, height_a], [0, height_a]], float
    )
    init_corners_b = np.array(
        [[0, 0], [width_b, 0], [width_b, height_b], [0, height_b]], float
    )

    transformed_corners_a = transform_corners(
        init_corners_a, b_target_onto_a_reference_h
    )
    transformed_corners_b = transform_corners(
        init_corners_b, b_target_onto_a_reference_h
    )

    polygon_a = create_polygon(init_corners_a)
    polygon_b = create_polygon(init_corners_b)
    transformed_polygon_a = create_polygon(transformed_corners_a)
    transformed_polygon_b = create_polygon(transformed_corners_b)

    intersection_in_b = compute_intersection(polygon_b, transformed_polygon_a)
    intersection_in_b = compute_intersection(polygon_a, transformed_polygon_b)

    mask_b = rasterize_polygon(intersection_in_b, (height_b, width_b))
    mask_a = rasterize_polygon(intersection_in_b, (height_a, width_a))

    return mask_a, mask_b
