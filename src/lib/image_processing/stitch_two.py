import cv2
import numpy as np


def stitch_two(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Returns the H matrix for the perspective based stitching
    of image B (sample) onto image A (reference),

    i.e. returns the matrix that transforms B to its closest matching overlap in image A

    uses the SIFT and other algorithms from opencv-contrib-python

    For simplicity, we use images A and B positioned at 0,0 (thus only need to use raw np.ndarray)
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    keypoints_A, descriptors_A = sift.detectAndCompute(A, None)
    keypoints_B, descriptors_B = sift.detectAndCompute(B, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_B, descriptors_A, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Need at least 4 good matches to find the homography
    if len(good_matches) < 4:
        raise ValueError("Not enough matches are found - {}/{}".format(len(good_matches), 4))

    # Extract location of good matches
    points_A = np.float32([keypoints_A[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_B = np.float32([keypoints_B[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    H, mask = cv2.findHomography(points_B, points_A, cv2.RANSAC, 5.0)

    return H