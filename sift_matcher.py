import cv2 as cv
import numpy as np


def order_points(pts):
    rect = np.zeros((2, 2), dtype="int32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmax(s)] - rect[0]
    return np.reshape(rect, (-1, 4))[0].tolist()


def sift_match(img1, img2, min_match=5, threshold=0.9):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = cv.BFMatcher().knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)

    if len(good) > min_match:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        h, w, c = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        dst = np.reshape(np.int32(dst), (4, 2))
        dst = order_points(dst)
        return tuple(dst)
    else:
        return None
