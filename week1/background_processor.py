import os
import cv2
import constants as C
from average_metrics import mapk
import numpy as np


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
    
    precision = truePositive / (truePositive + falsePositive)
    #print('Precision: ' + '{:.2f}'.format(precision))
    recall = truePositive / (truePositive + falseNegative)
    #print('Recall: ' + '{:.2f}'.format(recall))
    F1_measure = 2 * ((precision * recall) / (precision + recall))
    #print('F1-measure: ' + '{:.2f}'.format(F1_measure))
    return precision, recall, F1_measure



def backgroundRemoval(queryImage, filename):
    #Converting image to HSV and Lab
    queryImageHSV = cv2.cvtColor(queryImage, cv2.COLOR_BGR2HSV)
    queryImageLab = cv2.cvtColor(queryImage, cv2.COLOR_BGR2LAB)
    
    #Splitting in HSV and Lab channels
    h, s, v = cv2.split(queryImageHSV)
    L, a, b = cv2.split(queryImageLab)

    #Determining thresholds
    threshMinS, threshMaxS = 0, 60
    threshMinV, threshMaxV = 110, 255
    threshMinA, threshMaxA = 125, 255
    threshMinB, threshMaxB = 125, 255

    #Masks based on thresholds
    _, mask1S = cv2.threshold(s, threshMaxS, 255, cv2.THRESH_BINARY)
    _, mask2S = cv2.threshold(s, threshMinS, 255, cv2.THRESH_BINARY_INV)
    maskS = mask1S + mask2S
    _, mask1V = cv2.threshold(v, threshMaxV, 255, cv2.THRESH_BINARY)
    _, mask2V = cv2.threshold(v, threshMinV, 255, cv2.THRESH_BINARY_INV)
    maskV = mask1V + mask2V
    _, mask1A = cv2.threshold(a, threshMaxA, 255, cv2.THRESH_BINARY)
    _, mask2A = cv2.threshold(a, threshMinA, 255, cv2.THRESH_BINARY_INV)
    maskA = mask1A + mask2A
    _, mask1B = cv2.threshold(b, threshMaxB, 255, cv2.THRESH_BINARY)
    _, mask2B = cv2.threshold(b, threshMinB, 255, cv2.THRESH_BINARY_INV)
    maskB = mask1B + mask2B

    #Combining masks into a single mask
    mask = cv2.bitwise_not(intersect_matrices(cv2.bitwise_not(maskS), cv2.bitwise_not(maskV)))
    mask = cv2.bitwise_not(intersect_matrices(cv2.bitwise_not(mask), cv2.bitwise_not(maskA)))
    mask = cv2.bitwise_not(intersect_matrices(cv2.bitwise_not(mask), cv2.bitwise_not(maskB)))

    # Exporting mask as .png file
    cv2.imwrite(os.path.basename(filename).replace('jpg', 'png'), mask)

    
    # Mask evaluation
    annotationPath = filename.replace('jpg', 'png')
    
    if os.path.exists(annotationPath):
        annotation = cv2.imread(annotationPath)
        annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        precision, recall, F1_measure = evaluateMask(annotation, mask)
        
        return mask, precision, recall, F1_measure
    else:
        return mask, -1, -1, -1
