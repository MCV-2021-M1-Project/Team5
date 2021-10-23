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
from morphologicalOperations import thresholdImage, openingImage, closingImage, blackHat, topHat, morphologicalGradient, highpass

# convert xywh to box points
def convertBox(x, y, w, h):
    blx = x
    bly = y
    trx = x + w
    try1 = y + h
    return [blx, bly, trx, try1]

# gt_text_box to box points
def convertBox2(box):
    blx = box[0][0]
    bly = box[0][1]
    trx = box[2][0]
    try1 = box[1][1]
    return [blx, bly, trx, try1]

# Remove Boarder pixels
def removeBoarder(mask):
    h, w = mask.shape

    for i in range(0, h):
        for j in range(0, w):
            if i < 0.05 * h or j < 0.15 * w:
                mask[i, j] = 0
            elif i > 0.95 * h or j > 0.85 * w:
                mask[i, j] = 0
    return mask

#Get the most common pixel value from a masked grayscale image
def getMaskThreshold(gray, text_mask):
    colorSpace = "Gray"
    channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES[colorSpace][1:]

    # Get grayscale histogram of the mask
    hist = getHistogram2(gray, channels, text_mask, bins, colorRange)
    # plotHistogram(hist)

    # Convert histogram to simple list
    hist = [val[0] for val in hist]

    # Generate a list of indices
    indices = list(range(0, 256))

    # Descending sort-by-key with histogram value as key
    s = [(x, y) for y, x in sorted(zip(hist, indices), reverse=True)]

    # Index of highest peak in histogram
    peak = s[0][0]

    return peak

# Given an image return the mask of the text box
def getTextBox(image):
    # Convert image to grayscale color space
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.1, beta=1.4)

    # Use different stureturing elements for different sized images
    h, w, d = image.shape
    if h < 500 and w < 500:
        hat_structuring_element = (50, 10)
        opening_structuring_element = (1,1)
        closing_structuring_element = (15, 3)
    elif h < 1000 and w < 1000:
        hat_structuring_element = (50, 15)
        opening_structuring_element = (5,5)
        closing_structuring_element = (30, 8)
    elif h < 1500 and w < 1500:
        hat_structuring_element = (80, 25)
        opening_structuring_element = (5,5)
        closing_structuring_element = (50, 12)
    else:
        hat_structuring_element = (100, 30)
        opening_structuring_element = (5,5)
        closing_structuring_element = (50, 15)

    #Perform top hat & balck hat operation to extract text box and text
    black_hat = blackHat(gray, hat_structuring_element)
    top_hat = topHat(gray, hat_structuring_element)
    cv2.imshow("top_hat & black_hat", np.hstack([top_hat, black_hat]))

    top_hat_mask = thresholdImage(top_hat, 127)
    black_hat_mask = thresholdImage(black_hat, 127)

    # cv2.imshow("top_hat_mask & black_hat_mask", np.hstack([top_hat_mask, black_hat_mask]))

    # Combine top hat and black hat with an And gate
    hat2 = cv2.bitwise_or(top_hat_mask, black_hat_mask)

    # Replace pixel values at the boarder with zeros
    hat = removeBoarder(hat2.copy())

    # Remove noises from the mask
    text_mask = openingImage(hat, opening_structuring_element)
    text_mask = closingImage(text_mask, closing_structuring_element)
    # cv2.imshow("text_mask", np.hstack([hat, text_mask]))

    cv2.imshow("text_mask", text_mask)

    # Obtain the extract pixel value of the text box background
    peak = getMaskThreshold(gray, text_mask)

    # Threshold the image with the pixel value of the text box
    mask = cv2.inRange(gray, peak - 1, peak + 1)
    cv2.imshow("mask_thres", mask)

    # Remove noises from the mask
    mask = openingImage(mask, opening_structuring_element)
    mask = closingImage(mask, closing_structuring_element)

    cv2.imshow("final mask", mask)
    return mask

# Given image and maske return the rectangle coordinates x, y, w, h
def maskToRect(image, mask):
    output = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)

    # gradient = morphologicalGradient(mask, (3,3))
    # cv2.imshow("gradient", gradient)
    #
    # high = highpass(mask, 3)
    # cv2.imshow("highpass", high)

    ret, thresh = cv2.threshold(mask, 40, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # # show the images
    # cv2.imshow("Result", np.hstack([image, output]))
    # cv2.waitKey(0)
    return x, y, w, h

def getTextBoundingBox():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input image")
    ap.add_argument("-f", "--folder", help="path to input images")
    args, leftovers = ap.parse_known_args()

    with open("/Users/brian/Desktop/Computer Vision/M1/Project/qsd1_w2/text_boxes.pkl", 'rb') as reader:
        gt_boxes = pickle.load(reader)

    result = []
    counter = 0

    # load the image, convert it to grayscale, and display it to our
    # screen
    if args.folder is not None:
        filenames = [img for img in glob.glob(args.folder + "/*" + ".jpg")]
        filenames.sort()

        # Load images to a list
        images = []
        for i, img in enumerate(filenames):
            print("Painting:", i)
            image = cv2.imread(img)
            mask = getTextBox(image)
            x, y, w, h = maskToRect(image, mask)
            box = convertBox(x, y, w, h)

            gt_box = convertBox2(gt_boxes[i][0])
            iou = bbox_iou(box, gt_box)

            print("iou: ",iou)
            if box == [0, 0, 0, 0]:
                counter = counter + 1
            result.append(iou)

        print("Mean iou: ", sum(result) / len(result))
        print("Iou excluding failed cases: ", sum(result) / (len(result) - counter))
        print("Detection failed: ", counter)

    else:
        image = cv2.imread(args.image)
        mask = getTextBox(image)
        x, y, w, h = maskToRect(image, mask)

getTextBoundingBox()