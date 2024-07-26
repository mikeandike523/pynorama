from typing import List
from PIL import Image

import numpy as np

Image.MAX_IMAGE_PIXELS = None


class RGBAImage:

    def __init__(self, pixels: np.ndarray):
        self.pixels = pixels

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
    def from_file(cls, file_path: str) -> "RGBAImage":
        """Loads an RGBA image from a file and returns an instance of RGBAImage."""
        img = Image.open(file_path).convert("RGBA")
        pixels = np.array(img)
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
