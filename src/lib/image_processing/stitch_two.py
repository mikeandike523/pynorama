import cv2
import numpy as np

NUM_GOOD_MATCHES = 4
GOOD_MATCH_CUTOFF = 0.7
NUM_TREES=5
NUM_CHECKS=50
RANSAC_REPROJECTION_THRESHOLD=5.0

def stitch_two(A: np.ndarray, B: np.ndarray, exclude_fully_transparent=True) -> np.ndarray:
    """
    Returns the H matrix for the perspective based stitching
    of image B (sample) onto image A (reference).

    Arguments:
    A : ndarray - The reference image
    B : ndarray - The image to be transformed
    exclude_fully_transparent : boolean - Whether to exclude matches in fully transparent regions
    """

    # Create masks for fully transparent regions
    A_transparent_mask = A[:, :, 3] == 0 if A.shape[2] == 4 else None
    B_transparent_mask = B[:, :, 3] == 0 if B.shape[2] == 4 else None

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    keypoints_A, descriptors_A = sift.detectAndCompute(cv2.cvtColor(A, cv2.COLOR_BGRA2BGR), None)
    keypoints_B, descriptors_B = sift.detectAndCompute(cv2.cvtColor(B, cv2.COLOR_BGRA2BGR), None)

    if exclude_fully_transparent:
        # Filter out keypoints in transparent regions
        keypoints_A, descriptors_A = filter_keypoints(keypoints_A, descriptors_A, A_transparent_mask)
        keypoints_B, descriptors_B = filter_keypoints(keypoints_B, descriptors_B, B_transparent_mask)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=NUM_TREES)
    search_params = dict(checks=NUM_CHECKS)

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_B, descriptors_A, k=2)

    # Store all the good matches as per Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < GOOD_MATCH_CUTOFF * n.distance]

    if len(good_matches) < NUM_GOOD_MATCHES:
        raise ValueError(f"Not enough matches found - {len(good_matches)}/{NUM_GOOD_MATCHES}")

    # Extract location of good matches
    points_A = np.float32([keypoints_A[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_B = np.float32([keypoints_B[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    H, _ = cv2.findHomography(points_B, points_A, cv2.RANSAC, RANSAC_REPROJECTION_THRESHOLD)

    return H, len(good_matches)/len(matches)

def filter_keypoints(keypoints, descriptors, mask):
    """
    Filter out keypoints that lie in the masked (transparent) regions.
    """
    if mask is None:
        return keypoints, descriptors
    
    valid_keypoints = []
    valid_descriptors = []

    for keypoint, descriptor in zip(keypoints, descriptors):
        x, y = map(int, keypoint.pt)
        if not mask[y, x]:  # If the pixel is not transparent
            valid_keypoints.append(keypoint)
            valid_descriptors.append(descriptor)
        else:
            print(f"Skipping keypoint at ({x}, {y}) due to transparent region")

    return valid_keypoints, np.array(valid_descriptors)