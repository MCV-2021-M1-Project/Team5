# import the necessary packages
from __future__ import print_function
import argparse
import glob
import cv2
from text_processing import readTextFromFile
import pickle
import numpy as np
import constants as C
from average_metrics import bbox_iou
from matplotlib import pyplot as plt
import textdistance
from text_processing import imageToText
import pytesseract
from morphologicalOperations import thresholdImage, openingImage, closingImage, blackHat, topHat, morphologicalGradient, highpass
import platform

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

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

# xywh to pkl format
def convertBox3(x, y, w, h):
    tlx = x
    tly = y+h
    brx = x + w
    bry = y
    return [tlx, tly, brx, bry]

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
    hist = cv2.calcHist([gray], channels, text_mask, bins, colorRange)
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
    #cv2.imshow("top_hat & black_hat", np.hstack([top_hat, black_hat]))

    top_hat_mask = thresholdImage(top_hat, 127)
    black_hat_mask = thresholdImage(black_hat, 127)

    # #cv2.imshow("top_hat_mask & black_hat_mask", np.hstack([top_hat_mask, black_hat_mask]))

    # Combine top hat and black hat with an And gate
    hat2 = cv2.bitwise_or(top_hat_mask, black_hat_mask)

    # Replace pixel values at the boarder with zeros
    hat = removeBoarder(hat2.copy())

    # Remove noises from the mask
    text_mask = openingImage(hat, opening_structuring_element)
    text_mask = closingImage(text_mask, closing_structuring_element)
    # #cv2.imshow("text_mask", np.hstack([hat, text_mask]))

    #cv2.imshow("text_mask", text_mask)

    # Obtain the extract pixel value of the text box background
    peak = getMaskThreshold(gray, text_mask)

    # Threshold the image with the pixel value of the text box
    mask = cv2.inRange(gray, peak - 5, peak + 5)
    # cv2.imshow("Image After threshold", mask)

    # cv2.imshow("mask_thres", mask)

    # Remove noises from the mask
    mask = openingImage(mask, opening_structuring_element)
    mask = closingImage(mask, closing_structuring_element)

    #cv2.imshow("final mask", mask)
    return mask

# Given image and maske return the rectangle coordinates x, y, w, h
def maskToRect(image, mask):
    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    (numLabels, labels, boxes, centroids) = output

    boxes = np.delete(boxes, 0, 0)
    if boxes.shape[0] > 1:
        index = boxes.argmax(axis=0)[4]
        # print (index)
        x = boxes[index][0]
        y = boxes[index][1]
        w = boxes[index][2]
        h = boxes[index][3]
    else:
        if len(boxes) == 0: #If no box was found, set the 0,0 pixel
            x, y, w, h = 0, 0, 0, 0
        else:
            x = boxes[0][0]
            y = boxes[0][1]
            w = boxes[0][2]
            h = boxes[0][3]

    rows, cols = mask.shape
    new_mask = np.zeros((rows, cols), dtype="uint8")
    new_mask = cv2.rectangle(new_mask.copy(), (x, y), (x+w, y+h), 255, -1)

    image_masked = cv2.bitwise_and(image, image, mask=new_mask)

    cv2.rectangle(image_masked, (x, y), (x + w, y+h), (0, 255, 0), 2)

    # # show the images
    # cv2.imshow("Result", np.hstack([image, image_masked]))
    # cv2.waitKey(0)
    return new_mask, x, y, w, h

def getTextBoundingBoxAlone(image):
    mask = getTextBox(image)
    mask, x, y, w, h = maskToRect(image, mask)
    return mask, convertBox(x, y, w, h)

def getTextAlone(image):
    mask = getTextBox(image)
    mask, x, y, w, h = maskToRect(image, mask)
    if w > 0 and h > 0:
        img_cropped = image[y:y + h, x:x + w]
    else:
        img_cropped = image
    box = convertBox(x, y, w, h)
    return img_cropped, mask, box

def getTextBoundingBox(imageInput, folder = None, boxes_pkl = None):
    # construct the argument parser and parse the arguments

    with open(boxes_pkl, 'rb') as reader:
        gt_boxes = pickle.load(reader)

    result = []
    counter = 0

    # load the image, convert it to grayscale, and display it to our
    # screen
    if folder is not None:
        filenames = [img for img in glob.glob(folder + "/*" + ".jpg")]
        filenames.sort()

        # Load images to a list
        images = []
        for i, img in enumerate(filenames):
            print("Painting:", i)
            image = cv2.imread(img)
            mask = getTextBox(image)
            mask, x, y, w, h = maskToRect(image, mask)
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
        imageBGR = cv2.imread(imageInput)
        mask = getTextBox(imageBGR)
        mask, x, y, w, h = maskToRect(imageBGR, mask)
        return convertBox(x, y, w, h)

def test():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to input image")
    ap.add_argument("-f", "--folder", help="path to input images")
    args, leftovers = ap.parse_known_args()

    # with open("/Users/brian/Desktop/Computer Vision/M1/Project/qsd1_w3/text_boxes.pkl", 'rb') as reader:
    #     gt_boxes = pickle.load(reader)

    # with open("../datasets/qsd1_w3/text_boxes.pkl", 'rb') as reader:
    #     gt_boxes = pickle.load(reader)

    result = []
    distances = []
    counter = 0

    # load the image, convert it to grayscale, and display it to our
    # screen
    if args.folder is not None:
        filenames = [img for img in glob.glob(args.folder + "/*" + ".jpg")]
        filenames.sort()

        textnames = [text for text in glob.glob(args.folder + "/*" + ".txt")]
        textnames.sort()

        # Load images to a list
        images = []
        for i, img in enumerate(filenames):
            print("Painting:", i)
            image = cv2.imread(img)
            mask = getTextBox(image)
            mask, x, y, w, h = maskToRect(image, mask)
            image_masked = cv2.bitwise_and(image, image, mask=mask)
            if w > 0 and h > 0:
                img_cropped = image_masked[y:y + h, x:x + w]

                extractedtext = imageToText(img_cropped)

                with open(textnames[i], encoding="latin-1") as file:
                    lines = file.readlines()
                    painter_name, painting_name = readTextFromFile(lines[0])

                distance = textdistance.hamming.normalized_similarity(extractedtext, painter_name)
                distance1 = textdistance.hamming.normalized_similarity(extractedtext, painting_name)
                distance = max(distance, distance1)

            else:
                distance = 0

            print(distance)
            distances.append(distance)

        print("Mean distance: ", sum(distances) / len(distances))

    else:
        image = cv2.imread(args.image)
        mask = getTextBox(image)
        mask, x, y, w, h = maskToRect(image, mask)