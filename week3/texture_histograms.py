import os
import cv2
from statistics import mean
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
import constants as C
from average_metrics import getDistances
from background_processor import backgroundRemoval, intersect_matrices, findElementsInMask
from extractTextBox import getTextBoundingBoxAlone

#Select texture method to be applied
textureMethod = "DCT"

#LBP settings
radius = 8
n_points = 24
method = "nri_uniform"

#DCT settings
numCoef = 10

def getSingleTextureHistogram(image, channels, mask, bins, colorRange, sections = 1, maskPos = []):
    if textureMethod == "LBP":
        #Histogram settings
        bins = [150]
        colorRange = [0, 2+(n_points-1)*n_points] #Excluding non-uniform values (Non-inclusive end)
        
        lbp = local_binary_pattern(image, n_points, radius, method)
        lbp = np.uint16(lbp)
        textureInfo = lbp
        
    elif textureMethod == "DCT":
        imageConverted = np.float32(image)/255.0
        
    if sections <= 1:
        if textureMethod == "LBP":
            # Compute the histogram
            hist = cv2.calcHist([textureInfo], channels, mask, bins, colorRange)
            hist = cv2.normalize(hist, hist).flatten()
        elif textureMethod == "DCT":
            dctImg = cv2.dct(imageConverted)
            hist = dctCoefficients(dctImg[:5,:5])[:numCoef]
        return (abs(hist))
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
                if textureMethod == "LBP":
                    auxhist = cv2.calcHist([lbp], channels, sectionsMask, bins, colorRange)
                    auxHists.extend(cv2.normalize(auxhist, auxhist).flatten())
                elif textureMethod == "DCT":
                    imgSect = imageConverted[(hStart+sH*row):(sH*row + hStart + sH),(wStart+sW*column):(sW*column + wStart + sW)]
                    
                    dctImg = cv2.dct(imgSect)

                    # plt.imshow(cv2.cvtColor(imgSect, cv2.COLOR_GRAY2RGB))
                    # plt.axis("off")
                    # plt.show()
                    
                    # plt.imshow(cv2.cvtColor(dctImg, cv2.COLOR_GRAY2RGB))
                    # plt.axis("off")
                    # plt.show()
                    
                    plt.imshow(dctImg[:20,:20], cmap=cm.jet, interpolation = 'nearest')
                    plt.colorbar(shrink=0.5)
                    plt.show()
                    
                    hist = dctCoefficients(dctImg[:5,:5])[:numCoef]
                    # plotHistogram(hist)
                    auxHists.extend(abs(hist))
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
        # print(ddbb_histograms)
        results = getDistances(cv2.HISTCMP_BHATTACHARYYA, ddbb_histograms, queryHist)
        # sort the results
        allResults[0] = sorted([(v, k) for (k, v) in results.items()], reverse=False)
    return allResults

def dct2(image):
    return dct(dct(image, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def extractDctCoefficients(dctImage, coefH, coefW):
    coefList = []
    i = coefH
    j = coefW
    while i < dctImage.shape[0]:
        while j < dctImage.shape[1]:
            if dctImage[i][j] != 0:
                coefList.append(dctImage[i][j])
            j += 8
        j = coefW
        i += 8
    try:
        return mean(coefList)
    except:
        return 0

def dctCoefficients(dctImage):
    return np.concatenate([np.diagonal(dctImage[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctImage.shape[0], dctImage.shape[0])])
            
def plotHistogram(hist):
    # plot the histogram
    plt.figure()
    plt.xlabel("Coefficients")
    plt.ylabel("Values")
    plt.plot(hist)
    plt.xlim([150])

    plt.show()