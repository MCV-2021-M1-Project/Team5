import cv2

OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Chi-Squared", cv2.HISTCMP_CHISQR),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))

OPENCV_COLOR_SPACES =  {
    "Gray": [cv2.COLOR_BGR2GRAY, [0], None, [256], [0,256]],
    "Lab": [cv2.COLOR_BGR2LAB, [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256]],
    "HSV": [cv2.COLOR_BGR2HSV, [0, 1, 2], None, [8, 12, 3], [0, 179, 0, 255, 0, 255]],
    "YCrCb": [cv2.COLOR_BGR2YCrCb, [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256]],
    "RGB": [cv2.COLOR_BGR2RGB, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]]
    }
