# THE WHITE ROOM #

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
    plt.xlim([0, 16])

    plt.show()

query_image_folder = "../../Datasets/BBDD"

images = utils.loadAllImages(query_image_folder)
sections = 1

sList = []


uniformHistH = np.array([1/16]*16).astype("float32")

# print(uniformHistH)

for name, img in images.items():
    # img = img[round(img.shape[0]/10): round(img.shape[0]-img.shape[0]/10), round(img.shape[1]/10): round(img.shape[1]-img.shape[1]/10)]
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    sectionsMask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    sH = img.shape[0] // sections
    sW = img.shape[1] // sections
    hStart, wStart = 0, 0
    
    for row in range(sections):
        for column in range(sections):
            sectionsMask[(hStart+sH*row):(sH*row + hStart + sH),(wStart+sW*column):(sW*column + wStart + sW)] = 255
            imgSegment = cv2.bitwise_and(img, img, mask=sectionsMask)
            
            histH = cv2.calcHist([imgSegment], [0], sectionsMask, [16], [0, 255])
            histH = cv2.normalize(histH, histH).flatten()
            
            histS = cv2.calcHist([imgSegment], [1], sectionsMask, [16], [0, 255])
            histS = cv2.normalize(histS, histS).flatten()
            
            histV = cv2.calcHist([imgSegment], [2], sectionsMask, [8], [0, 255])
            histV = cv2.normalize(histV, histV).flatten()
            
            # print(histH)
            # print(histS)
            # print(histV)
            
            # plotHistogram(histS)
            # plotHistogram(histV)

            
            # plt.imshow(cv2.cvtColor(imgSegment, cv2.COLOR_HSV2RGB))
            # plt.axis("off")
            # plt.title("aux")
            # plt.show()
            
            sectionsMask[:,:] = 0
            
            uniformityH = cv2.compareHist(uniformHistH, histH, cv2.HISTCMP_INTERSECT)
            print(uniformityH)
            
            highS = sum(histS[13:16])
            lowS = sum(histS[0:4])
            
            lowV = sum(histV[0:5])

            sList.append([name, (50*uniformityH+10*highS-5*lowS-10*lowV)])

sList = sorted(sList, key=lambda x: x[1])
sList.reverse()
print(sList)

