import numpy as np

from .RGBAInfiniteCanvas import RGBAInfiniteCanvas
from .compute_overlapping_pixels import compute_overlapping_pixels
from .warp_without_cropping import warp_without_cropping
from .RGBAImage import RGBAImage
from .apply_h_matrix_to_point import apply_h_matrix_to_point


def mix_into_opaque(A: np.ndarray, B: np.ndarray):
    """
    Mixes two RGBA pixel arrays by weighted sum of their alpha values

    The RGB channels are mixed by weighted sum, florred to be value uint8, and then the alpha channel of the the result
    is set to entirely 255

    Edge cases are handled:

    If both alphas are 0, then the result is black

    """

    assert isinstance(A, np.ndarray) and isinstance(
        B, np.ndarray
    ), "Both inputs must be numpy arrays"

    assert A.shape == B.shape, "Both input arrays must have the same shape"

    assert (
        A.dtype == np.uint8 and B.dtype == np.uint8
    ), "Both input arrays must be of dtype np.uint8"

    assert A.ndim == 3 and B.ndim == 3, "Both input arrays must be 3D (RGBA)"

    assert (
        A.shape[2] == 4 and B.shape[2] == 4
    ), "Both input arrays must have 4 channels (RGBA)"

    rA, gA, bA, aA = [A[:, :, c].astype(float) for c in range(4)]
    rB, gB, bB, aB = [B[:, :, c].astype(float) for c in range(4)]

    denom_array = aA + aB
    denom_array[denom_array == 0] = 1
    r_result = np.floor((rA * aA + rB * aB) / denom_array).astype(np.uint8)
    g_result = np.floor((gA * aA + gB * aB) / denom_array).astype(np.uint8)
    b_result = np.floor((bA * aA + bB * aB) / denom_array).astype(np.uint8)
    return np.dstack([r_result, g_result, b_result, np.full(A.shape, 255)]).astype(
        np.uint8
    )


def compute_mse_overlap(A: np.ndarray, B: np.ndarray, H):
    # just in case

    A = A.copy()
    B = B.copy()

    mask_A, mask_B = compute_overlapping_pixels(A, B, H)

    A_masked = A.copy()
    A_masked[~mask_A, :] = 0

    B_masked = B.copy()
    B_masked[~mask_B, :] = 0

    init_B_corners = RGBAImage.from_pixels(B).corners()

    warped_B_corners = np.array(
        [apply_h_matrix_to_point(corner, H) for corner in init_B_corners], float
    )

    warped_B_tlc = warped_B_corners.min(axis=0)

    warped_B_tlc_rounded = np.round(warped_B_tlc).astype(int)

    tlc_x, tlc_y = warped_B_tlc_rounded

    mask_A_as_RGBA = np.dstack([np.where(mask_A, 0, 255).astype(np.uint8)] * 4)

    mask_B_as_RGBA = np.dstack([np.where(mask_B, 0, 255).astype(np.uint8)] * 4)

    mask_B_as_RGBA_warped = warp_without_cropping(mask_B_as_RGBA, H)

    B_masked_warped = warp_without_cropping(B_masked, H)

    mask_canvas = RGBAInfiniteCanvas()
    isolated_A_canvas = RGBAInfiniteCanvas()
    isolated_B_canvas = RGBAInfiniteCanvas()

    mask_canvas.place_pixel_array(mask_A_as_RGBA, 0, 0)
    mask_canvas.place_pixel_array(mask_B_as_RGBA_warped, tlc_x, tlc_y)

    isolated_A_canvas.place_pixel_array(A_masked, 0, 0)
    isolated_A_canvas.place_pixel_array(np.zeros_like(B_masked_warped), tlc_x, tlc_y)

    isolated_B_canvas.place_pixel_array(np.zeros_like(A_masked), 0, 0)
    isolated_B_canvas.place_pixel_array(B_masked_warped, tlc_x, tlc_y)

    mc_pixels = mask_canvas.canvas
    mc = mc_pixels[:, :, 3] != 0

    iAc_pixels = isolated_A_canvas.canvas
    iBc_pixels = isolated_B_canvas.canvas

    iAc_pixels[~mc, :] = 0
    iBc_pixels[~mc, :] = 0

    denom = np.count_nonzero(mc)

    if denom == 0:
        raise ZeroDivisionError("After transforming by H, B does not overlap with A")

    black_background = np.zeros_like(mc_pixels)
    black_background[~mc, 3] = 1

    iAc_pixels = mix_into_opaque(iAc_pixels.copy(), black_background)
    iBc_pixels = mix_into_opaque(iBc_pixels.copy(), black_background)

    iAc_pixels_scaled = iAc_pixels.astype(float) / 255
    iBc_pixels_scaled = iBc_pixels.astype(float) / 255
    square_errors = np.power((iAc_pixels_scaled - iBc_pixels_scaled), 2) * (255**2)
    r_se, g_se, b_se, _ = [square_errors[:, :, c] for c in range(4)]
    return np.sum(r_se + g_se + b_se) / (3 * denom)
