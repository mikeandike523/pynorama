from typing import Union

import humanfriendly
import numpy as np
import numpy.typing as npt

class PutItem:

    def __init__(self, x: int, y: int, data: np.ndarray):
        self.x = x
        self.y = y
        self.data = data


class Rect:

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains(self, x: int, y: int) -> bool:
        if x < self.x:
            return False
        if x >= self.x + self.width:
            return False
        if y < self.y:
            return False
        if y >= self.y + self.height:
            return False
        return True

    def intersects(self, other: "Rect") -> bool:
        return any(
            [
                self.contains(other.x, other.y),
                self.contains(other.x + other.width - 1, other.y),
                self.contains(other.x, other.y + other.height - 1),
                self.contains(other.x + other.width - 1, other.y + other.height - 1),
            ]
        ) or any(
            [
                other.contains(self.x, self.y),
                other.contains(self.x + self.width - 1, self.y),
                other.contains(self.x, self.y + self.height - 1),
                other.contains(self.x + self.width - 1, self.y + self.height - 1),
            ]
        )


class WindowAxisInfo:
    """
    A helper class used in the process of setting
    and retrieving rectangular windows of data from
    a larger 2D array

    When computing windows, each axis can be comptued independently,
    thus this class accounts for one axis only

    Attributes:
        big_start (int): The inclusive start index in the larger axis
                         Used to start the slicing in the main data
        big_end_excl (int): The exclusive end index in the larger axis
                            Used to end the slicing in the main data
        small_start (int): The inclusive start index in the smaller axis
                           Used to start the slicing in the window data
        small_end_excl (int): The exclusive end index in the smaller axis
                        Used to end the slicing in the window data
    """

    def __init__(self, big_start, big_end_excl, small_start, small_end_excl):
        self.big_start = big_start
        self.big_end_excl = big_end_excl
        self.small_start = small_start
        self.small_end_excl = small_end_excl

    @classmethod
    def compute(cls, size, t: int, window_size: int) -> "WindowAxisInfo":

        if t + window_size <= 0:
            return None
        if t >= size:
            return None

        big_start = max(0, t)
        big_end_excl = min(size, t + window_size)
        small_end_excl = (
            max(0, window_size - (t + window_size - size))
            if (t + window_size > size)
            else window_size
        )
        small_start = -t if t < 0 else 0
        return cls(big_start, big_end_excl, small_start, small_end_excl)


def compute_2D_windowing_axis_data(W: int, H: int, x: int, y: int, wW: int, wH: int):
    return {
        # axis 1 in numpy
        "x": WindowAxisInfo.compute(W, x, wW),
        # axis 0 in numpy
        "y": WindowAxisInfo.compute(H, y, wH),
    }


def copy_and_set_window(
    data: np.ndarray, x: int, y: int, window_data: np.ndarray
) -> None:
    H, W = data.shape[:2]
    wH, wW = window_data.shape[:2]
    result = data.copy()
    axis_data = compute_2D_windowing_axis_data(W, H, x, y, wW, wH)
    axis_data_x = axis_data["x"]
    axis_data_y = axis_data["y"]
    if axis_data_x is None or axis_data_y is None:
        return result
    result[
        axis_data_y.big_start : axis_data_y.big_end_excl,
        axis_data_x.big_start : axis_data_x.big_end_excl,
    ] = window_data[
        axis_data_y.small_start : axis_data_y.small_end_excl,
        axis_data_x.small_start : axis_data_x.small_end_excl,
    ]
    return result


# NEW VERSION -- SPARSE

# in the future, instead of looping through the geometry of each PutItem,
# we can speed it up by putting that geometry in a quadtree or kdtree


class Infinite2DCanvas:
    def __init__(self, dtype: Union[npt.DTypeLike, None] = None):
        if dtype is None:
            raise ValueError("Data type (dtype) must be specified")

        self.dtype = dtype
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

        self.put_items = []

    def get_width(self) -> int:
        return self.max_x - self.min_x + 1

    def get_height(self) -> int:
        return self.max_y - self.min_y + 1

    def __compute_bounds(self, x: int, y: int) -> None:
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)

    def __compute_bounds_rect(self, x: int, y: int, width: int, height: int) -> None:
        last_x = x + width - 1
        last_y = y + height - 1
        self.__compute_bounds(x, y)
        self.__compute_bounds(last_x, last_y)

    def put(self, pixels: np.ndarray, x: int = 0, y: int = 0) -> None:
        self.__compute_bounds_rect(x, y, pixels.shape[1], pixels.shape[0])
        self.put_items.append(PutItem(x, y, pixels.copy()))

    def get_footprint(self) -> str:
        total_bytes = 0
        for item in self.put_items:
            pixels_bytes = item.data.nbytes
            total_bytes += pixels_bytes
        return humanfriendly.format_size(total_bytes, binary=True)

    def get(
        self, x: int, y: int, width: int = 0, height: int = 0, default_value=0
    ) -> np.ndarray:
        relevant_items_in_order = [
            item
            for item in self.put_items
            if Rect(item.x, item.y, item.data.shape[1], item.data.shape[0]).intersects(
                Rect(x, y, width, height)
            )
        ]
        result = np.full((height, width), default_value, dtype=self.dtype)
        for item in relevant_items_in_order:
            dx = item.x - x
            dy = item.y - y
            window_data = item.data
            result = copy_and_set_window(result, dx, dy, window_data)
        return result

    def to_array(self) -> np.ndarray:
        return self.get(0, 0, self.get_width(), self.get_height())
