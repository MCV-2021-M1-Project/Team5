# import the necessary packages
from __future__ import print_function
import argparse
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input image")
ap.add_argument("-f", "--folder", help="path to input images")
# ap.add_argument("-op", "--operation", tyr=str, required=True, help="morphological operation to use")
args, leftovers = ap.parse_known_args()

def MSERImage(image, gray):
	# # detect MSER keypoints in the image
	# detector = cv2.MSER_create()
	# kps = detector.detect(gray)
	#
	# print("# of keypoints: {}".format(len(kps)))
	#
	# # loop over the keypoints and draw them
	# for kp in kps:
	# 	r = int(0.5 * kp.size)
	# 	(x, y) = np.int0(kp.pt)
	# 	cv2.circle(image, (x, y), r, (0, 255, 255), 2)
	#
	# # show the image
	# cv2.imshow("Images", np.hstack([image, image]))
	# cv2.waitKey(0)

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

def thresholdImage(gray, threslod):
	_, mask = cv2.threshold(gray, threslod, 255, cv2.THRESH_BINARY)
	# cv2.imshow('mask', mask)
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


def mask_to_rect(image, mask):
	output = cv2.bitwise_and(image, image, mask=mask)
	ret, thresh = cv2.threshold(mask, 40, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# if len(contours) != 0:
		# find the biggest countour (c) by the area
		# c = max(contours, key=cv2.contourArea)
		# x, y, w, h = cv2.boundingRect(c)
	x, y, w, h = cv2.boundingRect(thresh)

	# draw the biggest contour (c) in green
	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the images
	cv2.imshow("Result", np.hstack([image, output]))

	cv2.waitKey(0)


# load the image, convert it to grayscale, and display it to our
# screen
if args.folder is not None:
	filenames = [img for img in glob.glob(args.folder + "/*"+ ".jpg")]
	filenames.sort()

	# Load images to a list
	images = []
	for img in filenames:
		image = cv2.imread(img)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# cv2.imshow("Original", image)

		# erodeImage(gray, 3)
		# dilateImage(gray, 3)
		# openingImage(gray, (100,100))
		# closingImage(gray, (100,100))
		# MSERImage(image, gray)
		# morphologicalGradient(gray, (50,50))

		h, w, d = image.shape
		print(h, w, d)
		black_hat = blackHat(gray, (int(round(h/20)), int(round(h/50))))
		top_hat = topHat(gray, (int(round(h/20)), int(round(h/50))))

		top_hat_mask = thresholdImage(top_hat, 170)
		black_hat_mask = thresholdImage(black_hat, 170)
		mask = cv2.bitwise_or(top_hat_mask, black_hat_mask)

		closing = closingImage(mask, (70, 30))
		opening = openingImage(closing, (50,10))
		closing2 = closingImage(opening, (70, 30))

		mask_to_rect(image, closing2)

		cv2.waitKey(0)
else:
	image = cv2.imread(args.image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("Original", image)

	# erodeImage(gray, 3)
	# dilateImage(gray, 3)
	# openingImage(gray, (100,100))
	# closingImage(gray, (100,100))
	# MSERImage(image, gray)
	morphologicalGradient(gray, (50,50))

	h, w, d = image.shape
	print(h, w, d)
	black_hat = blackHat(gray, (int(round(h/20)), int(round(h/50))))
	top_hat = topHat(gray, (int(round(h/20)), int(round(h/50))))

	top_hat_mask = thresholdImage(top_hat, 170)
	black_hat_mask = thresholdImage(black_hat, 170)
	mask = cv2.bitwise_or(top_hat_mask, black_hat_mask)

	closing = closingImage(mask, (70, 30))
	opening = openingImage(closing, (50,10))
	closing2 = closingImage(opening, (70, 30))

	mask_to_rect(image, closing2)

	cv2.waitKey(10)