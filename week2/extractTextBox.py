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
from morphologicalOperations import thresholdImage, openingImage, closingImage, blackHat, topHat

def convertBox(x, y, w, h):
	blx = x
	bly = y
	trx = x + w
	try1 = y + h
	return [blx, bly, trx, try1]

def convertBox2(box):
	blx = box[0][0]
	bly = box[0][1]
	trx = box[2][0]
	try1 = box[1][1]
	return [blx, bly, trx, try1]

# Given an image return the mask of the text box
def getTextBox(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	h, w, d = image.shape
	hat_structuring_element = (int(round(h / 20)), int(round(h / 50)))
	print(h, w, d)

	#Perform morphological operations
	black_hat = blackHat(gray, hat_structuring_element)
	top_hat = topHat(gray, hat_structuring_element)
	# black_hat = blackHat(gray, (50, 15))
	# top_hat = topHat(gray, (50, 15))

	top_hat_mask = thresholdImage(top_hat, 170)
	black_hat_mask = thresholdImage(black_hat, 170)
	text_mask = cv2.bitwise_or(top_hat_mask, black_hat_mask)
	cv2.imshow("text_mask", text_mask)
	text_mask = openingImage(text_mask, (5, 5))
	text_mask = closingImage(text_mask, (30, 5))
	cv2.imshow("text_mask2", text_mask)

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

	# Threshold the image with the pixel value of the text box
	mask = cv2.inRange(gray, index_of_highest_peak - 2, index_of_highest_peak + 2)
	# cv2.imshow("mask", mask)

	opening = openingImage(mask, (10, 10))
	closing = closingImage(opening, (30, 5))
	return closing

# Given image and maske return the rectangle coordinates x, y, w, h
def maskToRect(image, mask):
	output = cv2.bitwise_and(image, image, mask=mask)
	x, y, w, h = cv2.boundingRect(mask)

	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the images
	cv2.imshow("Result", np.hstack([image, output]))
	cv2.waitKey(0)
	return x, y, w, h

def getTextBoundingBox():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", help="path to input image")
	ap.add_argument("-f", "--folder", help="path to input images")
	# ap.add_argument("-op", "--operation", tyr=str, required=True, help="morphological operation to use")
	args, leftovers = ap.parse_known_args()

	with open("/Users/brian/Desktop/Computer Vision/M1/Project/qsd1_w2/text_boxes.pkl", 'rb') as reader:
		gt_boxes = pickle.load(reader)

	# print(gt_boxes[0][0][0])
	# print(len(gt_boxes))
	# print(type(gt_boxes))

	result = []

	# load the image, convert it to grayscale, and display it to our
	# screen
	if args.folder is not None:
		filenames = [img for img in glob.glob(args.folder + "/*" + ".jpg")]
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

getTextBoundingBox()