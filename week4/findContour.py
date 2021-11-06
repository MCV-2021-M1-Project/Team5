import cv2
import imutils
import numpy as np
import pytesseract
from denoise_image import denoinseImage

image = cv2.imread("/Users/brian/Desktop/Computer Vision/M1/Project/qsd1_w4/00002.jpg")
image = denoinseImage(image)

# load the query image, compute the ratio of the old height
# to the new height, clone it, and resize it

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# perform a blackhat morphological operation that will allow
# us to reveal dark regions (i.e., text) on light backgrounds
# (i.e., the license plate itself)
rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
whitehat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKern)
cv2.imshow("Blackhat", blackhat)
cv2.imshow("Whitehat", whitehat)

text = pytesseract.image_to_string(whitehat)
print(text)



# # next, find regions in the image that are light
# squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
# light = cv2.threshold(light, 0, 255,
#                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow("Light Regions", light)
#
# # compute the Scharr gradient representation of the blackhat
# # image in the x-direction and then scale the result back to
# # the range [0, 255]
# gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
#                   dx=1, dy=0, ksize=-1)
# gradX = np.absolute(gradX)
# (minVal, maxVal) = (np.min(gradX), np.max(gradX))
# gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
# gradX = gradX.astype("uint8")
# cv2.imshow("Scharr", gradX)
#
# # blur the gradient representation, applying a closing
# # operation, and threshold the image using Otsu's method
# gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
# gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
# thresh = cv2.threshold(gradX, 0, 255,
#                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
# cv2.imshow("Grad Thresh", thresh)
#
# # perform a series of erosions and dilations to clean up the
# # thresholded image
# thresh = cv2.erode(thresh, None, iterations=2)
# thresh = cv2.dilate(thresh, None, iterations=2)
# cv2.imshow("Grad Erode/Dilate", thresh)
#
# # find contours in the thresholded image and sort them by
# # their size in descending order, keeping only the largest
# # ones
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                         cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
#
# # initialize the license plate contour and ROI
# lpCnt = None
# roi = None
#
# # loop over the license plate candidate contours
# for c in cnts:
#     # compute the bounding box of the contour and then use
#     # the bounding box to derive the aspect ratio
#     (x, y, w, h) = cv2.boundingRect(c)
#     ar = w / float(h)

cv2.waitKey(0)

gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 60, 120)
cv2.imshow("contour", edged)
#
# minLineLength = 200
# maxLineGap = 20
# lines = cv2.HoughLinesP(edged,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
# for x in range(0, len(lines)):
#     for x1,y1,x2,y2 in lines[x]:
#         cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imshow('hough',image)
# cv2.waitKey(0)
#
#
# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.015 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 4)
        cv2.imshow("image", image)

cv2.waitKey(0)
