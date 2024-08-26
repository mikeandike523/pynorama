from typing import List

import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
from .PixelDownsampler import PixelDownsampler


Image.MAX_IMAGE_PIXELS = None


class RGBAImage:

    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels.copy()

    def get_width(self) -> int:
        """Returns the width of the image (number of columns in the array)."""
        return self.pixels.shape[1]

    def get_height(self) -> int:
        """Returns the height of the image (number of rows in the array)."""
        return self.pixels.shape[0]

    @classmethod
    def check_pixels(cls, pixels: np.ndarray) -> bool:
        assert pixels.ndim == 3, "Image must be in RGB format"
        assert pixels.shape[2] == 4, "Image must be in RGBA format"
        assert pixels.dtype == np.uint8, "Image pixel values must be of type np.uint8"
        assert np.all(pixels >= 0) and np.all(
            pixels <= 255
        ), "Image pixel values must be in the range [0, 255]"

    @classmethod
    def from_pixels(cls, pixels: np.ndarray) -> "RGBAImage":
        cls.check_pixels(pixels)
        return cls(pixels)

    @classmethod
    def from_any_uint8_pixels(cls, pixels: np.ndarray) -> "RGBAImage":
        """
        Takes in image data with 1-4 channels and adds data as needed to make it the RGBA format

        One channel -> RGBA image with RGB identical alpha channel
        Two channels -> Fills the B channel with all 0 and A channel with all 255
        Three channels -> RGB image and A channel is all 255
        Four channels -> RGBA image (unchanged)

        Single channel data is squeezed as necessary before being converted to a consisten dimension
        This can be relevant in some cases where other algorithms
        tend to add a trivial dimension to single-channel images to keep them consistent with multi channel images


        Squeezing also helps in the case wher emore complex algorithms,
        especially those that work with multidimensional data,
        can introduce trivial extra dimensions to maintain consistent dimensionality
        """

        if not pixels.dtype == np.uint8:
            raise ValueError("Image pixel values must be of type np.uint8")

        pixels = pixels.squeeze()

        if pixels.ndim == 0:

            raise ValueError(
                """Input data is not image data.
                             It is a scalar.
                             """
            )

        if pixels.ndim == 1:

            raise ValueError(
                """
                             Input data is not image data. It is a vector.
                             Cannot be converted to image
                             as it is unknown if it should be horizontal or vertical.
                             """
            )

        if pixels.ndim == 2:
            R = pixels.copy()
            G = pixels.copy()
            B = pixels.copy()
            A = np.full(R.shape, 255, dtype=np.uint8)
            return cls(np.dstack((R, G, B, A)))

        if pixels.ndim == 3:
            existing_channels = [pixels[:, :, c] for c in range(pixels.shape[2])]

            while len(existing_channels) < 3:
                existing_channels.append(
                    np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.uint8)
                )

            while len(existing_channels) < 4:
                existing_channels.append(
                    np.full(pixels.shape[0:2], 255, dtype=np.uint8)
                )
            return cls(np.dstack(existing_channels))

        raise ValueError(
            f"Input data is not image data. It has {pixels.ndim} dimensions."
        )

    @classmethod
    def from_file(cls, file_path: str, downsampling=1.0) -> "RGBAImage":
        """Loads an RGBA image from a file and returns an instance of RGBAImage."""
        img = Image.open(file_path).convert("RGBA")
        pixels = PixelDownsampler(img.width, img.height, downsampling).downsample(
            np.asarray(img)
        )
        return cls.from_pixels(pixels)

    def corners(self) -> List[np.ndarray]:
        # A = topleft
        # B = topright
        # C = bottomright
        # D = bottomleft
        A = (0, 0)
        B = (self.get_width(), 0)
        C = (self.get_width(), self.get_height())
        D = (0, self.get_height())
        return [np.array(P, dtype=float) for P in [A, B, C, D]]

    def remove_background_otsu(
        self, blur_kernel_sigma1=1, blur_kernel_sigma2=2, morph_kernel_size=2
    ):
        """
        Returns a copy of this image with the background deleted
        (made black AND transparent, i.e. R=G=B=A=0)
        Using Otsu's method (enhanced with a Difference of Gaussians step)
        """
        # Convert RGBA to grayscale for thresholding
        grayscale_image = cv2.cvtColor(self.pixels, cv2.COLOR_RGBA2GRAY)

        # Apply two Gaussian blurs with different sigma values
        blurred_image1 = gaussian_filter(grayscale_image, sigma=blur_kernel_sigma1)
        blurred_image2 = gaussian_filter(grayscale_image, sigma=blur_kernel_sigma2)

        # Compute the Difference of Gaussians (DoG)
        dog_image = blurred_image1 - blurred_image2

        # Apply Otsu's thresholding to find the optimal threshold value
        _, binary_mask = cv2.threshold(
            dog_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((round(morph_kernel_size), round(morph_kernel_size)), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Convert binary mask to boolean for masking
        binary_mask_bool = binary_mask != 0

        # Create an empty image for the result with the same shape as the original image
        result_image = np.zeros_like(self.pixels)

        # Reinsert copies of each channel of the source pixel data with the Otsu threshold mask applied
        for channel in range(4):  # Iterate over RGBA channels
            result_image[:, :, channel] = self.pixels[:, :, channel] * binary_mask_bool

        # Return the result as a new RGBAImage instance
        return RGBAImage(result_image)

    def to_greyscale(self) -> "RGBAImage":
        R, G, B, A = (self.pixels[:, :, i] for i in range(4))
        combined = 1 / 3 * (R.astype(float) + G.astype(float) + B.astype(float))
        grey = combined.astype(np.uint8)
        return RGBAImage(np.dstack((grey, grey, grey, A)).astype(np.uint8))

    def gaussian_blurred(self, sigma: float) -> "RGBAImage":
        if sigma == 0:
            return RGBAImage(self.pixels)
        blurred_image = gaussian_filter(self.pixels, sigma=sigma)
        return RGBAImage(blurred_image)
