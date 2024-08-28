from enum import Enum
from warnings import warn

import cv2


import numpy as np


class SupportedResamplingAlgorithm(Enum):
    """
    A collection of the supported resampling algorithms
    for downsampling and upsampling images
    """

    LINEAR = cv2.INTER_LINEAR
    NEAREST = cv2.INTER_NEAREST
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA


class PixelDownsampler:
    """
    A class that provides a utility to downsample and upsample
       RGBA images with the following features:
       1. Uses rounding instead of flooring to determin the exact size
       of downsampled images
       2. Restores the original size of the image when upsampling
       3. Helps with consistency in resampling algorithms
    """

    def __init__(
        self,
        init_w,
        init_h,
        factor=1.0,
        interpolation_down=SupportedResamplingAlgorithm.AREA,
        interpolation_up=SupportedResamplingAlgorithm.LINEAR,
    ):
        """
        Arguments:
        init_w (int): The initial width of the image. Will be restored when upsampling
        init_h (int): The initial height of the image. Will be restored when upsampling
        factor (float): The downsampling factor
        (e.g. 2.0 means shrink to half the size / quarter the area)


        Remarks:
        In many applications, it is necessary to downsample and image, transform it,
        and then upsample it again.

        However, using plain `cv2.resize` can lead to mistakes and inconveniences
            such as:
            1. Not using the same interpolation method on downsampling and upsampling
            2. Getting an image that is a few pixels too large or too small
               after upsampling
               in the case where originally downsampling by a factor that does not
               evenly divide the original dimensions

        Additionally, when computing the size of the downsampledimage,
        we use rounding instead of flooring
        This helps remove bias towards shrinking image features during downsampling
        Nevertheless, this should be negligable unless the factor is very large
        """

        self.factor = factor

        self.interpolation_up = interpolation_up
        self.interpolation = interpolation_down

        self.init_w = init_w
        self.init_h = init_h

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

    def downsample(self, original: np.ndarray) -> np.ndarray:
        """
        Downsamples the input image using the specified resampling algorithm.

        Parameters:
        original (np.ndarray): The original image to be downsampled.
        It should be a 2D or 3D numpy array.

        Returns:
        np.ndarray: The downsampled image. The shape of the returned array will be
        (scaled_w, scaled_h),
        where scaled_w and scaled_h are the rounded down dimensions
        of the original image after applying the downsampling factor.

        Note:
        - If the downsampling factor is 1.0, the original image is returned
        without any modifications.
        - The resampling algorithm used for downsampling
        is specified by the 'interpolation_down' attribute
        of the PixelDownsampler instance.
        """
        if self.factor == 1.0:
            return original.copy()
        return cv2.resize(
            original,
            (self.scaled_w, self.scaled_h),
            interpolation=self.interpolation_down.value,
        )

    def upsample(self, downsampled: np.ndarray) -> np.ndarray:
        """
        Upsamples the input image to its original size using the specified
        resampling algorithm.

        Parameters:
        downsampled (np.ndarray): The downsampled image to be upsampled.
        It should be a 2D or 3D numpy array.

        Returns:
        np.ndarray: The upsampled image. The shape of the returned array will be
        (init_w, init_h),
        where init_w and init_h are the original dimensions of the image.

        Note:
        - If the downsampling factor is 1.0, the original image is returned
        without any modifications.
        - The resampling algorithm used for upsampling
        is specified by the 'interpolation_up' attribute
        of the PixelDownsampler instance.
        """
        if self.factor == 1.0:
            return downsampled.copy()
        return cv2.resize(
            downsampled,
            (self.init_w, self.init_h),
            interpolation=self.interpolation_up.value,
        )
