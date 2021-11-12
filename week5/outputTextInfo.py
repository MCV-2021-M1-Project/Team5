import cv2
import glob
import math
import statistics
import extractTextBox
import text_processing
from denoise_image import denoinseImage
from background_processor import backgroundRemoval, cleanerV, cleanerH, intersect_matrices
import textdistance
import pickle
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

#Open query image folder
query_image_folder = "../../Datasets/qsd1_w5"
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

for i, inputImage in enumerate(images):
    print('Processing image: ', filenames[i])
    filename = filenames[i]

    queryImage = denoinseImage(inputImage)
    
    #Converting image to Greyscale
    queryImageGray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
    
    #Draw contours
    contours = cv2.Canny(queryImageGray,50,160)
    
    plt.imshow(cv2.cvtColor(contours, cv2.COLOR_GRAY2RGB))
    plt.axis("off")
    plt.show()
    
    maskCleanV, maskCleanH = np.copy(contours), np.copy(contours)
    maskCleanV, maskCleanH = cleanerV(maskCleanV), cleanerH(maskCleanH)
    
    maskClean = intersect_matrices(maskCleanV, maskCleanH)
    
    plt.imshow(cv2.cvtColor(maskClean, cv2.COLOR_GRAY2RGB))
    plt.axis("off")
    plt.show()
    
    contours = cv2.Canny(maskClean,20,400)
    
    plt.imshow(cv2.cvtColor(contours, cv2.COLOR_GRAY2RGB))
    plt.axis("off")
    plt.show()
    
    lines = cv2.HoughLinesP(contours, 1, np.pi/180, 50, minLineLength = 150, maxLineGap = 20)
    angles = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(queryImage,(x1,y1),(x2,y2),(0,0,255),10)
        
        angle = math.atan2(y2-y1, x2-x1) * (180.0 / math.pi)
        if angle < 0:
            if angle < -45:
                angle += 90
        else:
            if angle > 45:
                angle -= 90
        angles.append(angle)
        
    print(statistics.median(angles))

    plt.imshow(cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


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