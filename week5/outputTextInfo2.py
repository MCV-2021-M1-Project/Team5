import cv2
import glob
import math
import statistics
import extractTextBox
import text_processing
from denoise_image import denoinseImage
from rotation import findAngle, rotate
# from background_processor import backgroundRemoval, cleanerV, cleanerH, intersect_matrices
from background_processor import backgroundFill, backgroundRemoval, findElementsInMask, crop_minAreaRect
import textdistance
import pickle
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

with open('/Users/brian/Desktop/Computer Vision/M1/Project/qsd1_w5/frames.pkl', 'rb') as f:
    gt_frames = pickle.load(f)

#Open query image folder
query_image_folder = "/Users/brian/Desktop/Computer Vision/M1/Project/qsd1_w5"
filenames = [img for img in glob.glob(query_image_folder + "/*"+ ".jpg")]
filenames.sort()

#Load images to a list
images = []
for img in filenames:
    n = cv2.imread(img)
    images.append(n)

masks = []
TextBoxPickle = []
start, end = [], []
distance_list = []
angle_error = []

for i, inputImage in enumerate(images):
    print("\n")
    print('Processing image: ', filenames[i])
    filename = filenames[i]

    print("gt_frames: ", gt_frames[i])

    queryImage = denoinseImage(inputImage)

    maskFill = backgroundFill(queryImage)
    # angle = findAngle(maskFill)
    # rotatedImage = rotate(queryImage, angle)
    # rotatedMaskFill = rotate(maskFill, angle)

    backgroundMask, precision, recall, F1_measure, IoU = backgroundRemoval(maskFill, filename)

    croppedImages = []
    frames = []
    contours, hierarchy = cv2.findContours(backgroundMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)

        # croppedImages.append(crop_minAreaRect(queryImage, rect))
        #
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)

        if rect[2] <= 0:
            angle = 90 + abs(rect[2])
        else:
            angle = 180 - rect[2]

        # if rect[2] < 45:
        #     angle = 180 - rect[2]
        # elif rect[2] == 180:
        #     angle = rect[2] - 90
        # else:
        #     angle = rect[2]
        # frame = [angle, box]
        print("Angle: ", angle)


        # frames.append(frame)


    # backgroundMask, precision, recall, F1_measure = background_processor.backgroundRemoval(queryImage, filename)
    # print("F1_measure: " + str(F1_measure))

    # contours, hierarchy = cv2.findContours(backgroundMask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]
    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # print("Rotation: " + str(rect[2]))
    # cv2.drawContours(queryImage, [box], 0, (0, 0, 255), 6)
    # plt.imshow(queryImage)
    # plt.show()

    # crop = background_processor.crop_minAreaRect(queryImage, rect)
    # plt.imshow(crop)
    # plt.show()