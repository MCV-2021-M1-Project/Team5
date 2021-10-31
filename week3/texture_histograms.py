import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct, idct
import constants as C
from average_metrics import getDistances
from background_processor import backgroundRemoval, intersect_matrices, findElementsInMask
from extractTextBox import getTextBoundingBoxAlone

#LBP settings
radius = 8
n_points = 24
method = "nri_uniform"

def getSingleTextureHistogram(image, channels, mask, bins, colorRange, sections = 1, maskPos = []):
    #Histogram settings
    bins = [150]
    colorRange = [0, 2+(n_points-1)*n_points] #Excluding non-uniform values (Non-inclusive end)
    
    lbp = local_binary_pattern(image, n_points, radius, method)
    lbp = np.uint16(lbp)
    
    # dct = dct2(image)
    # print(image.shape)
    # print(dct.shape)
    # print(dct)
    
    if sections <= 1:
        # Compute the histogram
        hist = cv2.calcHist([lbp], channels, mask, bins, colorRange)
        hist = cv2.normalize(hist, hist).flatten()
        return hist
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
        auxHists = []
        for row in range(sections):
            for column in range(sections):
                sectionsMask[(hStart+sH*row):(sH*row + hStart + sH),(wStart+sW*column):(sW*column + wStart + sW)] = 255
                if mask is not None:
                    sectionsMask = intersect_matrices(sectionsMask, mask)
                auxhist = cv2.calcHist([lbp], channels, sectionsMask, bins, colorRange)
                # plotHistogram(cv2.normalize(auxhist, auxhist).flatten())
                auxHists.extend(cv2.normalize(auxhist, auxhist).flatten())
                sectionsMask[:,:] = 0
        return auxHists

def getTextureHistogram(image, channels, masks, startMasks, endMasks, bins, colorRange, sections = 1):
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
            histograms.append(getSingleTextureHistogram(image, channels, mask, bins, colorRange, sections, maskPos))
        return histograms
    else:
        return getSingleTextureHistogram(image, channels, None, bins, colorRange, sections)

def getTextureHistograms(folderPath, sections = 1):
    """
    Returns a dict that contains texture histograms for each ddbb image

    :param folderPath: relative path to the images to process

    :return: 1- Dictionary with the texture histograms for all the images loaded ({'imageRelativePath': [histogram]})
    """ 
    ddbb_texture_histograms = {}
    
    for img in filter(lambda el: el.find('.jpg') != -1, os.listdir(folderPath)):
        filename = folderPath + '/' + img
        image = cv2.imread(filename)
        
        # Changing color space
        aux = cv2.cvtColor(image, C.OPENCV_COLOR_SPACES["Gray"][0])
        
        channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES["Gray"][1:]

        hist = getSingleTextureHistogram(aux, channels, mask, bins, colorRange, sections, [])

        ddbb_texture_histograms[filename] = hist
        
    return ddbb_texture_histograms

def getTextureHistogramForQueryImage(queryImage, backgroundMasks, startMasks, endMasks, sections = 1):    
    # Change to the color space that is going to be used to compare histograms
    queryImageColorSpace = cv2.cvtColor(queryImage, C.OPENCV_COLOR_SPACES["Gray"][0])
    channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES["Gray"][1:]
 
    queryHist = getTextureHistogram(queryImageColorSpace, channels, backgroundMasks, startMasks, endMasks, bins, colorRange, sections)
    
    return queryHist

def compareTextureHistograms(queryHist, ddbb_histograms):
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

def dct2(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def plotHistogram(hist):
    # plot the histogram
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Bins")
    plt.ylabel("% of Pixels")
    plt.plot(hist)
    plt.xlim([150])

    plt.show()