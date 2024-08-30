import os
from typing import Callable, Optional

import numpy as np
from lib.import_helpers.from_disconnected_import import from_disconnected_import

# assuming live-image-viewer is a sibling of the root folder of this project
# which is typical for any engineer with multiple related projects and cusotm utilities


def get_send_image_function() -> Optional[Callable[[str,np.ndarray], None]]:

    rf = os.path.dirname(os.path.abspath(__file__))
    if not os.path.normpath(rf).startswith(os.environ["HOME"]):
        raise ValueError(
            "Cannot find live-image-viewer directory when running this program in a folder not beneath HOME"
        )

    while ".git" not in os.listdir(rf):
        rf = os.path.dirname(rf)
        if os.path.normpath(rf) == os.environ["HOME"]:
            raise ValueError(
                "This program is not running contained in a valid git repository"
            )
    send_image = from_disconnected_import(
        os.path.join(rf, "..", "live-image-viewer"),
        "live_image_viewer",
        ["pysrc", "client"],
        "send_image",
    )
    return send_image
