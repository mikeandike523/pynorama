from typing import List, Tuple, Union
import numpy as np


def apply_h_matrix_to_point(
    point: Union[np.ndarray, Tuple[float, float], List[float]],
    H: np.ndarray,
    epsilon=1e-9,
) -> np.ndarray:
    point = np.array(point).astype(float).squeeze()
    if point.ndim != 1 or point.shape[0] != 2:
        raise ValueError("Point must be a 2D point (x, y)")
    x, y = point
    point_homogeneous = np.array([x, y, 1], dtype=float)
    transformed_point = np.matmul(H, point_homogeneous)
    new_x, new_y, new_w = transformed_point
    if np.abs(new_w) < epsilon:
        raise ValueError("resulting w is too small, check the H matrix")
    scaled_x = new_x / new_w
    scaled_y = new_y / new_w
    return np.array([scaled_x, scaled_y])
