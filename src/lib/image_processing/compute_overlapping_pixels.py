import numpy as np
from shapely.geometry import Polygon
from rasterio import features
from typing import Tuple, List
from .apply_h_matrix_to_point import apply_h_matrix_to_point


def create_polygon(corners: np.ndarray) -> Polygon:
    """Create a Shapely polygon from given corners."""
    return Polygon(corners)


def transform_corners(corners: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Transform corners using the given homography matrix."""
    return np.array([apply_h_matrix_to_point(point, H) for point in corners], float)


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
    A: np.ndarray, B: np.ndarray, H: np.ndarray
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
    HA, WA = A.shape[:2]
    HB, WB = B.shape[:2]

    B_target_onto_A_reference_H = H
    A_target_onto_B_reference_H = np.linalg.inv(H)

    init_corners_A = np.array([[0, 0], [WA, 0], [WA, HA], [0, HA]], float)
    init_corners_B = np.array([[0, 0], [WB, 0], [WB, HB], [0, HB]], float)

    transformed_corners_A = transform_corners(
        init_corners_A, A_target_onto_B_reference_H
    )
    transformed_corners_B = transform_corners(
        init_corners_B, B_target_onto_A_reference_H
    )

    polygon_A = create_polygon(init_corners_A)
    polygon_B = create_polygon(init_corners_B)
    transformed_polygon_A = create_polygon(transformed_corners_A)
    transformed_polygon_B = create_polygon(transformed_corners_B)

    intersection_in_B = compute_intersection(polygon_B, transformed_polygon_A)
    intersection_in_A = compute_intersection(polygon_A, transformed_polygon_B)

    mask_B = rasterize_polygon(intersection_in_B, (HB, WB))
    mask_A = rasterize_polygon(intersection_in_A, (HA, WA))

    return mask_A, mask_B
