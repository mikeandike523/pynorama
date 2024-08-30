# this is the rare case where the process of importing a module may fail
# this is where the benifit of imporst at runtime
# (as opposed to compile, build, or bundle time)
# adds convenience
# as if the dev knows that this improt will fail ina  certain situation,
# they can just not do it
# akin to nodejs async dynamic `import` function

import traceback
import warnings

import numpy as np

from .get_send_image_function import get_send_image_function

opt_end_image = get_send_image_function()


def send_image(image_path: str, pixels: np.ndarray) -> None:
    if opt_end_image is not None:
        try:
            opt_end_image(image_path, pixels)
        except Exception as e:
            warnings.warn(f"Failed to send image: {str(e)}")
            traceback.print_exc()
    else:
        warnings.warn("Failed to send image: live-image-viewer library not found")


__ALL__ = ["send_image"]
