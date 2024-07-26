import cv2
import numpy as np


def warp_by_corners(img, warped_corners):
    """
    Warps an image based on the given warped corners
    and expands the image with a specified background color.

    Parameters:
    img (numpy.ndarray): The input image to be warped.
    warped_corners (numpy.ndarray): The four corners of the destination image.

    Returns:
    numpy.ndarray: The warped image with the expanded background.
    """

    background_color = (0, 0, 0, 0)

    # Get the size of the input image
    height, width = img.shape[:2]

    # Define the original corners of the image
    original_corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)

    # Compute the homography matrix
    H, _ = cv2.findHomography(original_corners, warped_corners)

    # Compute the size of the output image
    warped_corners = warped_corners.reshape(-1, 2)
    min_x = np.min(warped_corners[:, 0])
    max_x = np.max(warped_corners[:, 0])
    min_y = np.min(warped_corners[:, 1])
    max_y = np.max(warped_corners[:, 1])

    output_width = int(np.ceil(max_x - min_x))
    output_height = int(np.ceil(max_y - min_y))

    # Adjust the homography matrix to account for the offset
    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    H_adjusted = np.dot(translation_matrix, H)

    # Create an RGBA image with the specified background color
    background_image = np.zeros((output_height, output_width, 4), dtype=np.uint8)
    background_image[:, :, :] = background_color

    # Warp the image
    warped_image = cv2.warpPerspective(img, H_adjusted, (output_width, output_height))

    # If the input image is not already RGBA, convert it
    if img.shape[2] == 3:
        warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2BGRA)

    # Blend the warped image onto the background image
    mask = warped_image[:, :, 3] > 0  # Non-transparent pixels
    background_image[mask] = warped_image[mask]

    return background_image
