import enum
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import constants as C
from average_metrics import getDistances
from background_processor import backgroundRemoval, intersect_matrices, findElementsInMask
from extractTextBox import getTextBoundingBoxAlone

def getSingleColorHistogram(image, channels, mask, bins, colorRange, sections = 1, maskPos = []):
    if sections <= 1:
            # Compute the histogram with color space passed as argument
            queryHist = cv2.calcHist([image], channels, mask, bins, colorRange)
            queryHist = cv2.normalize(queryHist, queryHist).flatten()
            return queryHist
    else:
        sectionsMask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
        sH = image.shape[0] // sections
        sW = image.shape[1] // sections
        hStart, wStart = 0, 0
        if len(maskPos) > 0:
            sectionsMaskAux = np.zeros((maskPos[1][0] - maskPos[0][0], maskPos[1][1] - maskPos[0][1]), dtype="uint8")
            sH = sectionsMaskAux.shape[0] // sections
            sW = sectionsMaskAux.shape[1] // sections
            hStart = maskPos[0][0]
            wStart = maskPos[0][1]
            # print(f'Size img: {image.shape}, auxMaskShape: {sectionsMaskAux.shape}, regSize: {sH}-{sW}, {hStart}-{wStart}')
        auxHists = []
        for row in range(sections):
            for column in range(sections):
                sectionsMask[(hStart+sH*row):(sH*row + hStart + sH),(wStart+sW*column):(sW*column + wStart + sW)] = 255
                if mask is not None:
                    sectionsMask = intersect_matrices(sectionsMask, mask)
                # plt.imshow(sectionsMask, cmap='gray')
                # plt.show()
                # cv2.waitKey(0)
                auxhist = cv2.calcHist([image], channels, sectionsMask, bins, colorRange)
                auxHists.extend(cv2.normalize(auxhist, auxhist).flatten())
                sectionsMask[:,:] = 0
        return auxHists


def getColorHistogram(image, channels, masks, startMasks, endMasks, bins, colorRange, sections = 1):
    """
    Compute the histogram for a given image and with the specified arguments for the histogram.
    If sections is bigger than 1 the image will be splited into sections*sections before computing.
    """

    #Check the mask to figure out if there are more than one pictures in the image
    if len(masks) > 0:
        histograms = []
        for ind, mask in enumerate(masks):
            maskPos = []
            if len(startMasks) > 0:
                maskPos = [startMasks[ind], endMasks[ind]]
            histograms.append(getSingleColorHistogram(image, channels, mask, bins, colorRange, sections, maskPos))
        return histograms
    else:
        return getSingleColorHistogram(image, channels, None, bins, colorRange, sections)


def plotHistogram(hist):
    # plot the histogram
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Bins")
    plt.ylabel("% of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])

    plt.show()

def loadAllImages(folderPath):
    
    ddbb_images = {}
    
    for img in filter(lambda el: el.find('.jpg') != -1, os.listdir(folderPath)):
        filename = folderPath + '/' + img
        image = cv2.imread(filename)

        # Store the image as RGB for later plot
        ddbb_images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return ddbb_images

def getColorHistograms(folderPath, colorSpace, sections = 1):
    """
    Returns a dict that contain color histograms for each ddbb image

    :param folderPath: relative path to the images to process
    :param colorSpace: color space to compute the histogramas of the images

    :return: 1- Dictionary with the histograms for all the images loaded ({'imageRelativePath': [histogram]})
    """ 
    ddbb_color_histograms = {}
    
    for img in filter(lambda el: el.find('.jpg') != -1, os.listdir(folderPath)):
        filename = folderPath + '/' + img
        image = cv2.imread(filename)
        
        # Equalizing Saturation and Lightness via HSV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v, = cv2.split(image)
        eqV = cv2.equalizeHist(v)
        image = cv2.merge((h, s, eqV))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        
        # Changing color space
        aux = cv2.cvtColor(image, C.OPENCV_COLOR_SPACES[colorSpace][0])
        
        channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES[colorSpace][1:]

        hist = getSingleColorHistogram(aux, channels, None, bins, colorRange, sections, [])

        ddbb_color_histograms[filename] = hist
        
    return ddbb_color_histograms
    
def getColorHistogramForQueryImage(queryImage, colorSpace, backgroundMasks, startMasks, endMasks, sections = 1):        
    # Equalizing Saturation and Lightness via HSV
    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2HSV)
    h, s, v, = cv2.split(queryImage)
    eqV = cv2.equalizeHist(v)
    queryImage = cv2.merge((h, s, eqV))
    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_HSV2BGR)
    
    # Change to the color space that is going to be used to compare histograms
    queryImageColorSpace = cv2.cvtColor(queryImage, C.OPENCV_COLOR_SPACES[colorSpace][0])
    channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES[colorSpace][1:]
 
    # Compute the histogram with color space passed as argument
    queryHist = getColorHistogram(queryImageColorSpace, channels, backgroundMasks, startMasks, endMasks, bins, colorRange, sections)
    
    return queryHist

def compareColorHistograms(queryHist, ddbb_histograms):
    """
    Compare the histograms of ddbb_histograms with the one for queryImage and returns
    a dictionary of diferent methods

    :param queryHist: histogram of queryImage to search
    :param ddbb_histograms: dictionary with the histograms of the images where queryImage is going to be searched

    :return: Dictionary with all the distances for queryImage ordered
    """ 
    
    shapeQueryHist = np.shape(queryHist)

    allResults = {}
    #Compute the distance to DDBB images with Hellinger distance metric
    if len(shapeQueryHist) > 1:
        for idx, hist in enumerate(queryHist):
            results = getDistances(cv2.HISTCMP_BHATTACHARYYA, ddbb_histograms, hist)
            # sort the results
            allResults[idx] = sorted([(v, k) for (k, v) in results.items()], reverse=False)
    else:
        results = getDistances(cv2.HISTCMP_BHATTACHARYYA, ddbb_histograms, queryHist)
        # sort the results
        allResults[0] = sorted([(v, k) for (k, v) in results.items()], reverse=False)
        
    return allResults