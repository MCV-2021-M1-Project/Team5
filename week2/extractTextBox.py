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

#Get the most common pixel value from a masked gray image
def getMaskThreshold(gray, text_mask):
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
    peak = s[0][0]

    return peak

# Given an image return the mask of the text box
def getTextBox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1, beta=1.4)

    h, w, d = image.shape
    if h < 500 & w < 500:
        structuring_element = (30, 10)
        structuring_element2 = (3,3)
    elif h < 1000 & w < 1000:
        structuring_element = (50, 15)
        structuring_element2 = (3,3)
    elif h < 1500 & w < 1500:
        structuring_element = (80, 25)
        structuring_element2 = (5,5)
    else:
        structuring_element = (100, 30)
        structuring_element2 = (10,10)

    #Perform morphological operations
    black_hat = blackHat(gray, structuring_element)
    top_hat = topHat(gray, structuring_element)
    cv2.imshow("top_hat & black_hat", np.hstack([top_hat, black_hat]))

    top_hat_mask = thresholdImage(top_hat, 150)
    black_hat_mask = thresholdImage(black_hat, 150)

    cv2.imshow("top_hat_mask & black_hat_mask", np.hstack([top_hat_mask, black_hat_mask]))

    # top_hat_sum = np.sum(top_hat_mask)
    # black_hat_sum = np.sum(black_hat_mask)
    # print(top_hat_sum, black_hat_sum)
    # if top_hat_sum > black_hat_sum:
    #     hat = top_hat_mask
    # else:
    #     hat = black_hat_mask

    hat = cv2.bitwise_or(top_hat_mask, black_hat_mask)
    # text_mask = openingImage(hat, (3, 3))
    text_mask = closingImage(hat, (30, 8))
    # cv2.imshow("text_mask", np.hstack([hat, text_mask]))

    cv2.imshow("text_mask", text_mask)

    peak = getMaskThreshold(gray, text_mask)

    # Threshold the image with the pixel value of the text box
    mask = cv2.inRange(gray, peak - 1, peak + 1)

    mask = openingImage(mask, structuring_element2)
    mask = closingImage(mask, (15, 5))
    cv2.imshow("final mask", mask)
    return mask

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
            if iou == 0:
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