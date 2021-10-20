# import the necessary packages
import argparse
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
# ap.add_argument("-op", "--operation", tyr=str, required=True, help="morphological operation to use")
args = vars(ap.parse_args())

# apply erosion
def erodeImage(gray, iteration):
	eroded = cv2.erode(gray.copy(), None, iterations=iteration)
	cv2.imshow("Eroded {} times".format(iteration), eroded)
	cv2.waitKey(0)

def dilateImage(gray, iteration):
	eroded = cv2.dilate(gray.copy(), None, iterations=iteration)
	cv2.imshow("Dilate {} times".format(iteration), eroded)
	cv2.waitKey(0)

def openingImage(gray, kernel_size):
	# construct a rectangular kernel from the current size and then
	# apply an "opening" operation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
	opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	cv2.imshow("Opening: ({}, {})".format(
		kernel_size[0], kernel_size[1]), opening)
	cv2.waitKey(0)

def closingImage(gray, kernel_size):
	# construct a rectangular kernel from the current size and then
	# apply an "closing" operation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
	closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
	cv2.imshow("Closing: ({}, {})".format(
		kernel_size[0], kernel_size[1]), closing)
	cv2.waitKey(0)

def morphologicalGradient(gray, kernel_size):
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
	gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
	cv2.imshow("Gradient: ({}, {})".format(
		kernel_size[0], kernel_size[1]), gradient)
	cv2.waitKey(0)

def topHat(gray, kernel_size):
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
	tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
	cv2.imshow("Tophat", tophat)
	cv2.waitKey(0)


def blackHat(gray, kernel_size):
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
	blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
	cv2.imshow("Blackhat", blackhat)
	cv2.waitKey(0)

# load the image, convert it to grayscale, and display it to our
# screen
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)

# erodeImage(gray, 3)
# dilateImage(gray, 3)
# openingImage(gray, (100,100))
# closingImage(gray, (100,100))
# morphologicalGradient(gray, (50,50))
blackHat(gray, (100, 10))
topHat(gray, (100,10))