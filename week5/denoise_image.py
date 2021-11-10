import cv2
from matplotlib import pyplot as plt


def denoinseImage(image, gt = None):
    inputsNoise =cv2.Laplacian(image, cv2.CV_64F).var()
    if inputsNoise > 2000.0:
        gauss = cv2.GaussianBlur(image, (5, 5), 0)
        median = cv2.medianBlur(image, 3)
        gaussNoise =cv2.Laplacian(gauss, cv2.CV_64F).var()
        medianNoise =cv2.Laplacian(median, cv2.CV_64F).var()
        return median if medianNoise > gaussNoise else gauss
    else:
        return image
