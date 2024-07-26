from enum import Enum
from warnings import warn

import cv2

from .RGBAImage import RGBAImage

import numpy as np


class SupportedResamplingAlgorithm(Enum):
    LINEAR = cv2.INTER_LINEAR
    NEAREST = cv2.INTER_NEAREST
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA


class RGBADownsampler:
    """
    A class that can hold on to a downsampled version of an RGBA image
    to conserve memory and improve performance
    Allows the contained data to be mutated and upsampled later

    Facilitates upsampling back to the original image size,
    Normally, when directly downsampling and upsampling,
    dimensions can be off by a few pixels
    Storing the original size can prevent that

    Note: Due to rounding and matching dimensions, the true
    downsampling factors may differ slightly along each dimension, and may differ
    between upsampling and downsampling

    however this error is generally negligible in practice,
    and preservation of image shape is more important


    Images are typically downsampled upon loading
    and upsampled before saving

    Note:

    The default resampling algorithm used here is cubic interpolation

    This is because it is known to preserve the most qualatative meaningful information
    in the image with respect to human visual perception

    For other forms of data, such as heatmaps, nearest of linear is likely more appropriate
    """

    def __init__(
        self,
        image: np.ndarray,
        factor=1.0,
        interpolation=SupportedResamplingAlgorithm.CUBIC,
    ):
        self.factor = factor

        self.interpolation = interpolation

        RGBAImage.check_pixels(image)
        self.init_w = image.shape[1]
        self.init_h = image.shape[0]

        float_scaled_w = float(self.init_w) / factor
        float_scaled_h = float(self.init_h) / factor

        self.scaled_w = round(self.init_w / factor)
        self.scaled_h = round(self.init_h / factor)

        error_scaled_w = abs(float_scaled_w - self.scaled_w)
        error_scaled_h = abs(float_scaled_h - self.scaled_h)

        if error_scaled_w != 0 or error_scaled_h != 0:
            warn(
                f"""
Warning: Downsampling from {self.init_w}x{self.init_h} by factor {factor}
does not result in pixel perfect resizing.

The downsampled result has dimension that are rounded to the nearest integer

Error in width: {error_scaled_w} downsampled pixels
Error in height: {error_scaled_h} downsampled pixels
"""
            )

        # Add a passthrough condition to simplify the code at the call site
        if factor == 1.0:
            self.pixels = image.copy()

        else:
            self.pixels = cv2.resize(
                image, (self.scaled_w, self.scaled_h), interpolation=interpolation.value
            )

    def to_original_size(self) -> np.ndarray:
        if self.factor == 1.0:
            return self.pixels.copy()
        return cv2.resize(
            self.pixels,
            (self.init_w, self.init_h),
            interpolation=self.interpolation.value,
        )
