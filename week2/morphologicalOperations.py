# import the necessary packages
from __future__ import print_function
import argparse
import glob
import cv2
import pickle
import numpy as np
import constants as C
from average_metrics import bbox_iou
from histogram_processing import getHistogram2, plotHistogram

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

def convertBox(x, y, w, h):
	tly = y + h
	tlx = x
	bry = y
	brx = x + w
	return [tly, tlx, bry, brx]

def convertBox2(box):
	tly = box[1][1]
	tlx = box[0][0]
	bry = box[0][1]
	brx = box[2][0]
	return [tly, tlx, bry, brx]

def getTextBox(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	h, w, d = image.shape
	# print(h, w, d)

	# black_hat = blackHat(gray, (int(round(h / 20)), int(round(h / 50))))
	# top_hat = topHat(gray, (int(round(h / 20)), int(round(h / 50))))
	black_hat = blackHat(gray, (50, 15))
	top_hat = topHat(gray, (50, 15))

	top_hat_mask = thresholdImage(top_hat, 170)
	black_hat_mask = thresholdImage(black_hat, 170)
	text_mask = cv2.bitwise_or(top_hat_mask, black_hat_mask)
	text_mask = openingImage(text_mask, (5, 5))
	text_mask = closingImage(text_mask, (30, 5))

	# cv2.imshow("text_mask", text_mask)

	colorSpace = "Gray"
	channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES[colorSpace][1:]

	hist = getHistogram2(gray, channels, text_mask, bins, colorRange)
	# plotHistogram(hist)

	# Convert histogram to simple list
	hist = [val[0] for val in hist]

	# Generate a list of indices
	indices = list(range(0, 256))

	# Descending sort-by-key with histogram value as key
	s = [(x, y) for y, x in sorted(zip(hist, indices), reverse=True)]

	# Index of highest peak in histogram
	index_of_highest_peak = s[0][0]

	mask = cv2.inRange(gray, index_of_highest_peak - 2, index_of_highest_peak + 2)
	# cv2.imshow("mask", mask)

	opening = openingImage(mask, (10, 10))
	closing = closingImage(opening, (30, 5))
	return closing

def maskToRect(image, mask):
	output = cv2.bitwise_and(image, image, mask=mask)
	x, y, w, h = cv2.boundingRect(mask)

	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the images
	cv2.imshow("Result", np.hstack([image, output]))
	cv2.waitKey(0)
	return x, y, w, h


with open("/Users/brian/Desktop/Computer Vision/M1/Project/qsd1_w2/text_boxes.pkl", 'rb') as reader:
	gt_boxes = pickle.load(reader)

# print(gt_boxes[0][0][0])
# print(len(gt_boxes))
# print(type(gt_boxes))


result = []

# load the image, convert it to grayscale, and display it to our
# screen
if args.folder is not None:
	filenames = [img for img in glob.glob(args.folder + "/*"+ ".jpg")]
	filenames.sort()

	# Load images to a list
	images = []
	for i, img in enumerate(filenames):
		image = cv2.imread(img)
		mask = getTextBox(image)
		x, y, w, h = maskToRect(image, mask)
		box = convertBox(x, y, w, h)

		gt_box = convertBox2(gt_boxes[i][0])
		iou = bbox_iou(box, gt_box)

		print("Painting:", i)
		# print(gt_boxes[i][0])
		print(box)
		print(gt_box)
		print(iou)
		result.append(iou)

	print(sum(result) / len(result))

else:
	image = cv2.imread(args.image)
	mask = getTextBox(image)
	x, y, w, h = maskToRect(image, mask)