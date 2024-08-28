"""
A utility to compute the Mean Squared Error (MSE) between two RGBA images
in the overlapping region after the second image
is transformed using a homography matrix
"""

import numpy as np

from .compute_overlapping_pixels import compute_overlapping_pixels
from .RGBAImage import RGBAImage
from .RGBAInfiniteCanvas import RGBAInfiniteCanvas
from .warp_without_cropping import warp_without_cropping
from .apply_h_matrix_to_point import apply_h_matrix_to_point
from .mix_into_opaque import mix_into_opaque


def compute_mse_overlap(image_a: np.ndarray, image_b: np.ndarray, homography):
    """
    A utility to compute the Mean Squared Error (MSE) between two RGBA images
    in the overlapping region after the second image
    is transformed using a homography matrix
    """

    image_a = image_a.copy()
    image_b = image_b.copy()

    mask_a, mask_b = compute_overlapping_pixels(image_a, image_b, homography)

    a_masked = image_a.copy()
    a_masked[~mask_a, :] = 0

    b_masked = image_b.copy()
    b_masked[~mask_b, :] = 0

    init_b_corners = RGBAImage.from_pixels(image_b).corners()

    warped_b_corners = np.array(
        [apply_h_matrix_to_point(corner, homography) for corner in init_b_corners],
        float,
    )

    warped_b_tlc = warped_b_corners.min(axis=0)

    warped_b_tlc_rounded = np.round(warped_b_tlc).astype(int)

    tlc_x, tlc_y = warped_b_tlc_rounded

    mask_a_as_rgba = np.dstack([np.where(mask_a, 0, 255).astype(np.uint8)] * 4)

    mask_b_as_rgba = np.dstack([np.where(mask_b, 0, 255).astype(np.uint8)] * 4)

    mask_b_as_rgba_warped = warp_without_cropping(mask_b_as_rgba, homography)

    b_masked_warped = warp_without_cropping(b_masked, homography)

    mask_canvas = RGBAInfiniteCanvas()
    isolated_a_canvas = RGBAInfiniteCanvas()
    isolated_b_canvas = RGBAInfiniteCanvas()

    mask_canvas.place_pixel_array(mask_a_as_rgba, 0, 0)
    mask_canvas.place_pixel_array(mask_b_as_rgba_warped, tlc_x, tlc_y)

    isolated_a_canvas.place_pixel_array(a_masked, 0, 0)
    isolated_a_canvas.place_pixel_array(np.zeros_like(b_masked_warped), tlc_x, tlc_y)

    isolated_b_canvas.place_pixel_array(np.zeros_like(a_masked), 0, 0)
    isolated_b_canvas.place_pixel_array(b_masked_warped, tlc_x, tlc_y)

    mc_pixels = mask_canvas.canvas
    mc = mc_pixels[:, :, 3] != 0

    iac_pixels = isolated_a_canvas.canvas
    ibc_pixels = isolated_b_canvas.canvas

    iac_pixels[~mc, :] = 0
    ibc_pixels[~mc, :] = 0

    denom = np.count_nonzero(mc)

    if denom == 0:
        raise ZeroDivisionError("After transforming by H, B does not overlap with A")

    black_background = np.zeros_like(mc_pixels)
    black_background[~mc, 3] = 1

    iac_pixels = mix_into_opaque(iac_pixels.copy(), black_background)
    ibc_pixels = mix_into_opaque(ibc_pixels.copy(), black_background)

    iAc_pixels_scaled = iac_pixels.astype(float) / 255
    iBc_pixels_scaled = ibc_pixels.astype(float) / 255
    square_errors = np.power((iAc_pixels_scaled - iBc_pixels_scaled), 2) * (255**2)
    r_se, g_se, b_se, _ = [square_errors[:, :, c] for c in range(4)]
    return np.sum(r_se + g_se + b_se) / (3 * denom)
