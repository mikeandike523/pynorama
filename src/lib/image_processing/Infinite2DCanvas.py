import numpy as np
import humanfriendly


class Infinite2DCanvas:
    def __init__(self, initial_width=0, initial_height=0, dtype=None):
        if dtype is None:
            raise ValueError("Data type (dtype) must be specified")

        # Initialize an empty canvas with the given initial dimensions and data type
        self.canvas = np.zeros((initial_height, initial_width), dtype=dtype)
        # Track the current bounds of the canvas
        self.min_x = 0
        self.min_y = 0
        self.max_x = initial_width
        self.max_y = initial_height
        self.dtype = dtype

    def __expand_canvas(self, new_min_x, new_min_y, new_max_x, new_max_y):
        # Calculate new dimensions
        new_width = new_max_x - new_min_x
        new_height = new_max_y - new_min_y

        # Create a new larger canvas
        new_canvas = np.zeros((new_height, new_width), dtype=self.dtype)

        # Calculate the offsets for copying the old canvas into the new canvas
        offset_x = self.min_x - new_min_x
        offset_y = self.min_y - new_min_y

        # Copy the old canvas into the new canvas
        old_height, old_width = self.canvas.shape
        new_canvas[
            offset_y : offset_y + old_height, offset_x : offset_x + old_width
        ] = self.canvas

        # Update the canvas and bounds
        self.canvas = new_canvas
        self.min_x = new_min_x
        self.min_y = new_min_y
        self.max_x = new_max_x
        self.max_y = new_max_y

    def put(self, pixels, x=0, y=0):
        # Determine the dimensions of the incoming pixel array
        pixel_height, pixel_width = pixels.shape

        # Determine the required dimensions of the canvas
        required_min_x = min(self.min_x, x)
        required_min_y = min(self.min_y, y)
        required_max_x = max(self.max_x, x + pixel_width)
        required_max_y = max(self.max_y, y + pixel_height)

        # Expand the canvas if necessary
        if (
            required_min_x < self.min_x
            or required_min_y < self.min_y
            or required_max_x > self.max_x
            or required_max_y > self.max_y
        ):
            self.__expand_canvas(
                required_min_x, required_min_y, required_max_x, required_max_y
            )

        # Calculate the positions on the new canvas
        canvas_x = x - self.min_x
        canvas_y = y - self.min_y

        # Place the pixel array onto the canvas
        self.canvas[
            canvas_y : canvas_y + pixel_height, canvas_x : canvas_x + pixel_width
        ] = pixels

    def get(self, x=0, y=0, width=0, height=0):


        # Determine the required dimensions of the canvas
        required_min_x = min(self.min_x, x)
        required_min_y = min(self.min_y, y)
        required_max_x = max(self.max_x, x + width)
        required_max_y = max(self.max_y, y + height)

        # Expand the canvas if necessary
        if (
            required_min_x < self.min_x
            or required_min_y < self.min_y
            or required_max_x > self.max_x
            or required_max_y > self.max_y
        ):
            self.__expand_canvas(
                required_min_x, required_min_y, required_max_x, required_max_y
            )

        # Calculate the positions on the new canvas
        canvas_x = x - self.min_x
        canvas_y = y - self.min_y

        return self.canvas[
            canvas_y : canvas_y + height, canvas_x : canvas_x + width
        ]

    def to_array(self):
        return self.canvas.copy()

    def get_footprint(self):
        size_in_bytes = self.canvas.nbytes
        return humanfriendly.format_size(size_in_bytes, binary=True)
