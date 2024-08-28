"""
A utility to mix together two RGBA pixel arrays by weighted sum of their alpha values
The resulting array is still RGBA, but the alpha channel is set to entirely 255
"""

import numpy as np


def mix_into_opaque(image_a: np.ndarray, image_b: np.ndarray):
    """
    Mixes two RGBA pixel arrays by weighted sum of their alpha values

    The RGB channels are mixed by weighted sum, floored to be value uint8,
    and then the alpha channel of the the result
    is set to entirely 255

    Edge cases are handled:

    If both alphas are 0, then the result is black

    """

    assert isinstance(image_a, np.ndarray) and isinstance(
        image_b, np.ndarray
    ), "Both inputs must be numpy arrays"

    assert image_a.shape == image_b.shape, "Both input arrays must have the same shape"

    assert (
        image_a.dtype == np.uint8 and image_b.dtype == np.uint8
    ), "Both input arrays must be of dtype np.uint8"

    assert (
        image_a.ndim == 3 and image_b.ndim == 3
    ), "Both input arrays must be 3D (RGBA)"

    assert (
        image_a.shape[2] == 4 and image_b.shape[2] == 4
    ), "Both input arrays must have 4 channels (RGBA)"

    r_a, g_a, b_a, a_a = [image_a[:, :, c].astype(float) for c in range(4)]
    r_b, g_b, b_b, a_b = [image_b[:, :, c].astype(float) for c in range(4)]

    denom_array = a_a + a_b
    denom_array[denom_array == 0] = 1
    r_result = np.floor((r_a * a_a + r_b * a_b) / denom_array).astype(np.uint8)
    g_result = np.floor((g_a * a_a + g_b * a_b) / denom_array).astype(np.uint8)
    b_result = np.floor((b_a * a_a + b_b * a_b) / denom_array).astype(np.uint8)
    return np.dstack(
        [r_result, g_result, b_result, np.full(image_a.shape, 255)]
    ).astype(np.uint8)
