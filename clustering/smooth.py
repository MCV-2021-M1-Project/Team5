import math
import statistics
import cv2
from matplotlib import pyplot as plt
import glob
import utils
import numpy as np

def plotHistogram(hist):
    # plot the histogram
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Bins")
    plt.ylabel("% of Pixels")
    plt.plot(hist)
    plt.xlim([0, 10])

    plt.show()

def dctCoefficients(dctImage):
    return np.concatenate([np.diagonal(dctImage[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctImage.shape[0], dctImage.shape[0])])

query_image_folder = "../../Datasets/BBDD"

images = utils.loadAllImages(query_image_folder)

sections = 8

sList = []

for name, img in images.items():
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img = np.float32(img)/255.0
    
    img = img[round(img.shape[0]/4): round(img.shape[0]-img.shape[0]/4), round(img.shape[1]/4): round(img.shape[1]-img.shape[1]/4)]
    
    
    
    sectionsMask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    sH = img.shape[0] // sections
    sW = img.shape[1] // sections
    hStart, wStart = 0, 0

    
    points = 0
    
    for row in range(sections):
        for column in range(sections):
            imgSegment = img[(hStart+sH*row):(sH*row + hStart + sH),(wStart+sW*column):(sW*column + wStart + sW)]            
            dctImg = cv2.dct(imgSegment)
            
            # histV = dctImg[0, 1:5]
            # histH = dctImg[1:5, 0]
            # histOthers = dctImg[1:5, 1:5]
            hist = abs(dctCoefficients(dctImg[:5,:5])[:10])
            hist = cv2.normalize(hist, hist).flatten()

            # print(hist)

            # plt.imshow(cv2.cvtColor(imgSegment, cv2.COLOR_HSV2RGB))
            # plt.axis("off")
            # plt.title("aux")
            # plt.show()
            
            sectionsMask[:,:] = 0
            
    
            points += (10*hist[0] - hist[5] - hist[8] - hist[9])
            
    # print("Points = " + str(points))
    sList.append([name, points])
    points = 0

sList = sorted(sList, key=lambda x: x[1])
sList.reverse()
print(sList)
