import os
import cv2
from matplotlib import pyplot as plt
import constants as C
from average_metrics import mapk
import numpy as np
import math
import statistics as stat
import morphologicalOperations as mo



def intersect_matrices(m1, m2):
        if not (m1.shape == m2.shape):
            return False
        intersect = np.where((m1 == m2), m1, 0)
        return intersect

def evaluateMask(gtMask, computedMask):    
    #Compute the score for a given mask
    # gtMask and computedMask size have to be the same

    truePositive = np.count_nonzero(intersect_matrices(gtMask, computedMask))
    falseNegative = np.count_nonzero(intersect_matrices(gtMask, cv2.bitwise_not(computedMask)))
    falsePositive = np.count_nonzero(intersect_matrices(cv2.bitwise_not(gtMask), computedMask))
    # trueNegative = np.count_nonzero(intersect_matrices(cv2.bitwise_not(gtMask), cv2.bitwise_not(computedMask)))

    if truePositive + falsePositive != 0:
        precision = truePositive / (truePositive + falsePositive)
        # print('Precision: ' + '{:.2f}'.format(precision))
        recall = truePositive / (truePositive + falseNegative)
    else:
        precision = 0
        recall = 0
    if precision + recall != 0:
        #print('Recall: ' + '{:.2f}'.format(recall))
        F1_measure = 2 * ((precision * recall) / (precision + recall))
        #print('F1-measure: ' + '{:.2f}'.format(F1_measure))
    else:
        F1_measure = 0
    return precision, recall, F1_measure

def sorting_func(lst):
  return np.linalg.norm(lst[0])


def findElementsInMask(mask):
    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    (numLabels, labels, boxes, centroids) = output
    # print('resultados de connected: ',numLabels, labels, boxes)

    # plt.imshow(mask, cmap='gray')
    # plt.show()
    # cv2.waitKey(0)
    
    start, end = [], []
    for box in boxes[1:]:
        start.append([box[1], box[0]])
        end.append([box[1] + box[3], box[0] + box[2]])
    
    numberElements = numLabels - 1

    # print(start)
    # print(end)
    # print('-----------------------')
    if len(start) > 0:
        zipped_lists = zip(start, end)
        sorted_pairs = sorted(zipped_lists, key=sorting_func )

        tuples = zip(*sorted_pairs)
        start, end = [ list(tuple) for tuple in  tuples]
    else:
        start = [[0, 0]]
        end = [[np.shape(mask)[0], np.shape(mask)[1]]]

    # print(start)
    # print(end)
    
    return numberElements, start, end

def cleanerV(mask):
    i = 1 #Column iterator
    j = 1 #Row iterator
    while i < mask.shape[1]:
        eraserActive = False
        while j < mask.shape[0]:
            if mask[j, i] == 255:
                if mask[j-1, i] == 255:
                    eraserStart = j
                else:
                    if eraserActive == False:
                        eraserActive = True
                        eraserStart = j
                    else:
                        mask[(eraserStart+1):j, i] = 255
                        eraserStart = j
            j += 1
        j = 0
        i += 1
    return mask

def cleanerH(mask):
    i = 1 #Column iterator
    j = 1 #Row iterator
    while j < mask.shape[0]:
        eraserActive = False
        while i < mask.shape[1]:
            if mask[j, i] == 255:
                if mask[j, i-1] == 255:
                    eraserStart = i
                else:
                    if eraserActive == False:
                        eraserActive = True
                        eraserStart = i
                    else:
                        mask[j, (eraserStart+1):i] = 255
                        eraserStart = i
            i += 1
        i = 0
        j += 1
    return mask


def backgroundRemoval(queryImage, filename):
    #Converting image to Greyscale
    queryImageGray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
    
    #Draw contours
    mask = cv2.Canny(queryImageGray,50,160)
    
    #Ensuring there are no open regions on edges (10px minimum)
    mask[0:10, :], mask[-10:, :], mask[:, 0:10], mask[:, -10:] = 0, 0, 0, 0
    
    #Creating clean mask
    maskCleanV, maskCleanH = np.copy(mask), np.copy(mask)
    maskCleanV, maskCleanH = cleanerV(maskCleanV), cleanerH(maskCleanH)
    
    maskClean = intersect_matrices(maskCleanV, maskCleanH)

    #Adding extra margins (+20%) to mask for improved morphology
    maskCleanExt = cv2.copyMakeBorder(maskClean, round(maskClean.shape[0]*0.2), round(maskClean.shape[0]*0.2), round(maskClean.shape[1]*0.2), round(maskClean.shape[1]*0.2), cv2.BORDER_CONSTANT, None, value=0)

    #Further cleaning using morphology
    #Closing with _ as structuring element
    kernelSize = (round(maskClean.shape[0]/8), round(maskClean.shape[1]/30))
    maskClosing = mo.closingImage(maskCleanExt, kernelSize)
    #Opening with tall rectangle as structuring element (removing labels, etc.)
    kernelSize = (round(maskClean.shape[0]/4), round(maskClean.shape[1]/12))
    maskOpening = mo.openingImage(maskClosing, kernelSize)

    #Removing extra margins
    maskFinal = maskOpening[round(maskClean.shape[0]*0.2):-round(maskClean.shape[0]*0.2), round(maskClean.shape[1]*0.2):-round(maskClean.shape[1]*0.2)]

    #Exporting mask as .png file
    cv2.imwrite(os.path.basename(filename).replace('jpg', 'png'), maskFinal)

    # plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title('Mask basic')
    # plt.show()

    # plt.imshow(cv2.cvtColor(maskClean, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title('Mask cleaned')
    # plt.show()
    
    # plt.imshow(cv2.cvtColor(maskClosing, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title('Mask closing')
    # plt.show()
    
    # plt.imshow(cv2.cvtColor(maskOpening, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title('Mask opening')
    # plt.show()
    
    # plt.imshow(cv2.cvtColor(maskFinal, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title('Final mask')
    # plt.show()

    # Mask evaluation
    annotationPath = filename.replace('jpg', 'png')
    
    if os.path.exists(annotationPath):
        annotation = cv2.imread(annotationPath)
        annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        precision, recall, F1_measure = evaluateMask(annotation, maskFinal)
        
        return maskFinal, precision, recall, F1_measure
    else:
        return maskFinal, -1, -1, -1
