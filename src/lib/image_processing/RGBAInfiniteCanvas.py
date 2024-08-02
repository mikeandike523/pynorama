import numpy as np
import humanfriendly


class RGBAInfiniteCanvas:
    def __init__(self, initial_width=0, initial_height=0):
        # Initialize an empty RGBA canvas with the given initial dimensions
        self.canvas = np.zeros((initial_height, initial_width, 4), dtype=np.uint8)
        # Track the current bounds of the canvas
        self.min_x = 0
        self.min_y = 0
        self.max_x = initial_width
        self.max_y = initial_height

    def expand_canvas(self, new_min_x, new_min_y, new_max_x, new_max_y):
        # Calculate new dimensions
        new_width = new_max_x - new_min_x
        new_height = new_max_y - new_min_y

        # Create a new larger canvas
        new_canvas = np.zeros((new_height, new_width, 4), dtype=np.uint8)

        # Calculate the offsets for copying the old canvas into the new canvas
        offset_x = self.min_x - new_min_x
        offset_y = self.min_y - new_min_y

        # Copy the old canvas into the new canvas
        old_height, old_width, _ = self.canvas.shape
        new_canvas[
            offset_y : offset_y + old_height, offset_x : offset_x + old_width, :
        ] = self.canvas

        # Update the canvas and bounds
        self.canvas = new_canvas
        self.min_x = new_min_x
        self.min_y = new_min_y
        self.max_x = new_max_x
        self.max_y = new_max_y

    def place_pixel_array(self, pixel_array, top_left_x=0, top_left_y=0):
        # Determine the dimensions of the incoming pixel array
        pixel_height, pixel_width, _ = pixel_array.shape

        # Determine the required dimensions of the canvas
        required_min_x = min(self.min_x, top_left_x)
        required_min_y = min(self.min_y, top_left_y)
        required_max_x = max(self.max_x, top_left_x + pixel_width)
        required_max_y = max(self.max_y, top_left_y + pixel_height)

        # Expand the canvas if necessary
        if (
            required_min_x < self.min_x
            or required_min_y < self.min_y
            or required_max_x > self.max_x
            or required_max_y > self.max_y
        ):
            self.expand_canvas(
                required_min_x, required_min_y, required_max_x, required_max_y
            )

        # Calculate the positions on the new canvas
        canvas_x = top_left_x - self.min_x
        canvas_y = top_left_y - self.min_y

        # Extract existing pixels in the target region
        existing = self.canvas[
            canvas_y : canvas_y + pixel_height,
            canvas_x : canvas_x + pixel_width,
            :,
        ]

        # Determine transparent regions in the incoming pixel array
        transparent_region = pixel_array[:, :, 3] == 0

        # Make a copy of the pixel array
        pixel_array_copy = pixel_array.copy()

        # Set the values of pixel_array_copy to the values of existing pixels in the transparent regions
        pixel_array_copy[transparent_region, :] = existing[transparent_region, :]

        # Place the pixel array onto the canvas
        self.canvas[
            canvas_y : canvas_y + pixel_height,
            canvas_x : canvas_x + pixel_width,
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

    def measure_pixel_memory_footprint(self):
        size_in_bytes = self.canvas.nbytes
        return humanfriendly.format_size(size_in_bytes, binary=True)
