# import the necessary packages
from __future__ import print_function

import cv2
import numpy as np

def MSERImage(image, gray):
    # # detect MSER keypoints in the image

    mser = cv2.MSER_create()
    vis = image.copy()
    regions, _ = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    cv2.imshow('image', vis)

    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    text_only = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('mask', mask)

def thresholdImage(gray, threshold):
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask

# apply erosion
def erodeImage(gray, iteration):
    eroded = cv2.erode(gray.copy(), None, iterations=iteration)
    # cv2.imshow("Eroded {} times".format(iteration), eroded)

    return eroded

def dilateImage(gray, iteration):
    dilated = cv2.dilate(gray.copy(), None, iterations=iteration)
    # cv2.imshow("Dilate {} times".format(iteration), dilated)

    return dilated

def openingImage(gray, kernel_size):
    # construct a rectangular kernel from the current size and then
    # apply an "opening" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("Opening: ({}, {})".format(
    # 	kernel_size[0], kernel_size[1]), opening)

    return opening

def closingImage(gray, kernel_size):
    # construct a rectangular kernel from the current size and then
    # apply an "closing" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Closing: ({}, {})".format(
    # 	kernel_size[0], kernel_size[1]), closing)

    return closing

def morphologicalGradient(gray, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow("Gradient: ({}, {})".format(
    # 	kernel_size[0], kernel_size[1]), gradient)
    return gradient

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

def topHat(gray, kernel_size):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    # cv2.imshow("Tophat", tophat)

    return tophat


def blackHat(gray, kernel_size):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    # cv2.imshow("Blackhat", blackhat)

    return blackhat