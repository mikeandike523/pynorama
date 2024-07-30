import numpy as np
from pympler import asizeof
import humanfriendly


class RGBAInfiniteCanvas:
    def __init__(self, initial_width=100, initial_height=100):
        # Initialize an empty RGBA canvas with the given initial dimensions
        self.canvas = np.zeros((initial_height, initial_width, 4), dtype=np.uint8)

    def expand_canvas(self, new_width, new_height):
        # Create a new larger canvas
        new_canvas = np.zeros((new_height, new_width, 4), dtype=np.uint8)
        # Copy the old canvas into the new canvas
        old_height, old_width, _ = self.canvas.shape
        new_canvas[:old_height, :old_width, :] = self.canvas
        self.canvas = new_canvas

    def place_pixel_array(self, pixel_array, top_left_x=0, top_left_y=0):
        # Determine the dimensions of the incoming pixel array
        pixel_height, pixel_width, _ = pixel_array.shape

        # Determine the required dimensions of the canvas
        required_width = top_left_x + pixel_width
        required_height = top_left_y + pixel_height

        # Expand the canvas if necessary
        if (
            required_width > self.canvas.shape[1]
            or required_height > self.canvas.shape[0]
        ):
            new_width = max(required_width, self.canvas.shape[1])
            new_height = max(required_height, self.canvas.shape[0])
            self.expand_canvas(new_width, new_height)

        existing = self.canvas[
            top_left_y : top_left_y + pixel_height,
            top_left_x : top_left_x + pixel_width,
            :,
        ]

        transparent_region = pixel_array[:, :, 3] == 0

        pixel_array_copy = pixel_array.copy()

        # make the values of pixel_array_copy
        # be the values of existing pixels in transparent region
        pixel_array_copy[transparent_region, :] = existing[transparent_region, :]

        # Place the pixel array onto the canvas
        self.canvas[
            top_left_y : top_left_y + pixel_height,
            top_left_x : top_left_x + pixel_width,
            :,
        ] = pixel_array_copy

    def get_canvas(self):
        return self.canvas

    def save_canvas(self, filename):
        from PIL import Image

        image = Image.fromarray(self.canvas, "RGBA")
        image.save(filename)

    def show_canvas(self):
        from PIL import Image

        image = Image.fromarray(self.canvas, "RGBA")
        image.show()

    def get_footprint_human_readable(self):
        size_in_bytes = asizeof.asizeof(self.canvas)
        return humanfriendly.format_size(size_in_bytes)
