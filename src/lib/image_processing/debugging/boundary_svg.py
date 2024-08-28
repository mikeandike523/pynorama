"""
A utility to create an SVG file showing the boundaries of each image
in an assembled panorama

Used primarily for debugging
"""


from typing import List
import numpy as np


def boundary_svg(boundaries: List[np.ndarray]) -> None:
    """
    Generates the text for an svg containing a sequence of 1 pixel wide
    closed polygons using the path element.
    """

    # 1. Calculate the bounding box of all the polygons
    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
    for boundary in boundaries:
        x_min = min(x_min, np.min(boundary[:, 0]))
        y_min = min(y_min, np.min(boundary[:, 1]))
        x_max = max(x_max, np.max(boundary[:, 0]))
        y_max = max(y_max, np.max(boundary[:, 1]))

    # 2. Generate the SVG
    width = x_max - x_min
    height = y_max - y_min

    svg_header = (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    )
    svg_footer = "</svg>"
    svg_content = []

    for boundary in boundaries:
        d = "M " + " L ".join(f"{x},{y}" for x, y in boundary) + " Z"
        svg_content.append(
            f'<path d="{d}" stroke="black" fill="none" stroke-width="1"/>'
        )

    return "\n".join([svg_header] + svg_content + [svg_footer])
