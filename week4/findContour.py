import cv2
import imutils
import numpy as np
from denoise_image import denoinseImage

image = cv2.imread("/Users/brian/Desktop/Computer Vision/M1/Project/qsd1_w4/00000.jpg")
image = denoinseImage(image)

# load the query image, compute the ratio of the old height
# to the new height, clone it, and resize it

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 60, 120)
cv2.imshow("contour", edged)

minLineLength = 200
maxLineGap = 20
lines = cv2.HoughLinesP(edged,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('hough',image)
cv2.waitKey(0)


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
