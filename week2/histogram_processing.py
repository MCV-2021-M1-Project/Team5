import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import constants as C
from average_metrics import mapk
from background_processor import backgroundRemoval, intersect_matrices, findElementsInMask
from extractTextBox import getTextBoundingBoxAlone

def getSingleHistogram(image, channels, mask, bins, colorRange, sections = 1):
    if sections <= 1:
            # Compute the histogram with color space passed as argument
            queryHist = cv2.calcHist([image], channels, mask, bins, colorRange)
            queryHist = cv2.normalize(queryHist, queryHist).flatten()
            return queryHist
    else:
        sectionsMask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
        sH = image.shape[0] // sections
        sW = image.shape[1] // sections
        auxHists = []
        for row in range(sections):
            for column in range(sections):
                sectionsMask[sH*row:(sH*row + sH),sW*column:(sW*column + sW)] = 255
                if mask is not None:
                    sectionsMask = intersect_matrices(sectionsMask, mask)
                auxhist = cv2.calcHist([image], channels, sectionsMask, bins, colorRange)
                auxHists.extend(cv2.normalize(auxhist, auxhist).flatten())
                sectionsMask[:,:] = 0
        return auxHists


def getHistogram(image, channels, mask, bins, colorRange, sections = 1, textBoxImage = None):
    """
    Compute the histogram for a given image and with the specified arguments for the histogram.
    If sections is bigger than 1 the image will be splited into sections*sections before computing.
    """

    #Check the mask to figure out if there are more than one pictures in the image
    if mask is None:
        if textBoxImage is not None:
            box = getTextBoundingBoxAlone(textBoxImage)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
            mask[box[1]:box[3],box[0]:box[2]] = 255
            mask = cv2.bitwise_not(mask)
            # plt.imshow(mask, cmap='gray')
            # plt.show()
            # cv2.waitKey(0)
        return getSingleHistogram(image, channels, mask, bins, colorRange, sections)
    else:
        # plt.imshow(mask, cmap='gray')
        # plt.show()
        # cv2.waitKey(0)
        elems, start, end = findElementsInMask(mask)
        # print(f'Elementos {elems}, inicios {start}, finales {end}')
        if elems > 1:
            histograms = []
            for num in range(elems):
                auxMask = np.zeros(mask.shape, dtype="uint8")
                print(f'Size: {np.shape(auxMask)}, Rango {start[num][0]}:{end[num][0]} - {start[num][1]}:{end[num][1]}')
                auxMask[start[num][0]:end[num][0],start[num][1]:end[num][1]] = 255
                if textBoxImage is not None:
                    res = cv2.bitwise_and(textBoxImage,textBoxImage,mask = auxMask)
                    plt.imshow(res)
                    plt.show()
                    cv2.waitKey(0)
                    box = getTextBoundingBoxAlone(res)
                    textMask = np.zeros(mask.shape, dtype="uint8")
                    textMask[box[1]:box[3],box[0]:box[2]] = 255
                    auxMask = cv2.bitwise_and(auxMask,auxMask,mask = cv2.bitwise_not(textMask))
                    plt.imshow(auxMask, cmap='gray')
                    plt.show()
                    cv2.waitKey(0)
                histograms.append(getSingleHistogram(image, channels, auxMask, bins, colorRange, sections))
            return histograms
        else:
            return getSingleHistogram(image, channels, mask, bins, colorRange, sections)
    
def getHistogram2(image, channels, mask, bins, colorRange, sections = 1):
    if sections <= 1:
        # Compute the histogram with color space passed as argument
            queryHist = cv2.calcHist([image], channels, mask, bins, colorRange)
            return queryHist
    else:
        sectionsMask = np.zeros((image.shape[0], image.shape[1]), dtype="uint8")
        sH = image.shape[0] // sections
        sW = image.shape[1] // sections
        auxHists = []
        for row in range(sections):
            for column in range(sections):
                sectionsMask[sH*row:(sH*row + sH),sW*column:(sW*column + sW)] = 255
                if mask is not None:
                    sectionsMask = intersect_matrices(sectionsMask, mask)
                auxhist = cv2.calcHist([image], channels, sectionsMask, bins, colorRange)
                sectionsMask[:,:] = 0
        return auxHists

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

def getImagesAndHistograms(folderPath, colorSpace, sections = 1):
    """
    Returns a dict that contain all the jpg images from folder path, and 
    other one with the corresponding histograms for each image

    :param folderPath: relative path to the images to process
    :param colorSpace: color space to compute the histogramas of the images

    :return: 1- Dictionary with all the images loaded in RGB format so the can be plotted ({'imageRelativePath': [RGB image]})
             1- Dictionary with the histograms for all the images loaded ({'imageRelativePath': [histogram]})
    """ 
    ddbb_images = {}
    ddbb_histograms = {}
    
    for img in filter(lambda el: el.find('.jpg') != -1, os.listdir(folderPath)):
        filename = folderPath + '/' + img
        image = cv2.imread(filename)
        
        # Denoising images using Gaussian Blur
        image = cv2.GaussianBlur(image,(3,3), 0)
        
        # Equalizing Saturation and Lightness via HSV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v, = cv2.split(image)
        eqV = cv2.equalizeHist(v)
        image = cv2.merge((h, s, eqV))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        
        # Changing color space
        aux = cv2.cvtColor(image, C.OPENCV_COLOR_SPACES[colorSpace][0])
        
        # Store the image as RGB for later plot
        ddbb_images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES[colorSpace][1:]

        hist = getHistogram(aux, channels, mask, bins, colorRange, sections)

        ddbb_histograms[filename] = hist
        
    return ddbb_images, ddbb_histograms



def compareHistograms(queryImage, colorSpace, mask_check, ddbb_histograms, filename, sections = 1, textBox = False):
    """
    Compare the histograms of ddbb_histograms with the one for queryImage and returns
    a dictionary of diferent methods

    :param queryImage: image to look for in ddbb_histograms
    :param colorSpace: color space to compute the histogramas of the query image
    :param mask_check: if true tried to remove the background from queryImage
    :param ddbb_histograms: dictionary with the histograms of the images where queryImage is going to be searched
    :param filename: if mask_check is true it's used to load the gt mask and compute the quality of the computed mask

    :return: 1- Dictionary with all the distances for queryImage ordered for different Methods (format: {'MethodName': [Distances...]})
             2- precsion of the mask computed if mask_check and filename has a png with the ground truth, -1 othewise
             3- recall of the mask computed if mask_check and filename has a png with the ground truth, -1 othewise
             4- f1-measure of the mask computed if mask_check and filename has a png with the ground truth, -1 othewise
    """ 
    originalImage = queryImage
    # Denoising query image using Gaussian blur
    queryImage = cv2.GaussianBlur(queryImage,(3,3), 0)

    # Apply mask if applicable
    backgroundMask = None
    if mask_check:
        backgroundMask, precision, recall, F1_measure = backgroundRemoval(queryImage, filename)
        # backgroundMask = cv2.imread(filename.replace('jpg','png'), cv2.IMREAD_GRAYSCALE)
        # precision = -1
        # recall = -1
        # F1_measure = -1
        
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
    if textBox:
        queryHist = getHistogram(queryImageColorSpace, channels, backgroundMask, bins, colorRange, sections, originalImage)
    else:
        queryHist = getHistogram(queryImageColorSpace, channels, backgroundMask, bins, colorRange, sections, None)
    
    shapeQueryHist = np.shape(queryHist)

    allResults = None
    if len(shapeQueryHist) > 1:
        allResults = []
        for i in range(shapeQueryHist[0]):
            allResults.append({})
    else:
        allResults = {}

    # loop over the comparison methods
    for (methodName, method) in C.OPENCV_METHODS:
        # initialize the results dictionary and the sort
        # direction
        reverse = False

        # if we are using the correlation or intersection
        # method, then sort the results in reverse order
        if methodName in ("Correlation", "Intersection"):
            reverse = True
        if len(shapeQueryHist) > 1:
            for idx, hist in enumerate(queryHist):
                results = getDistances(method, ddbb_histograms, hist)
                allResults[idx][methodName] = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
        else:
            results = getDistances(method, ddbb_histograms, queryHist)
            allResults[methodName] = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
        # sort the results
        
    if mask_check:
        return allResults, precision, recall, F1_measure
    else:
        return allResults, -1, -1, -1


def getDistances(comparisonMethod, baseImageHistograms, queryImageHistogram):
    # loop over the index
    results = {}
    for (k, hist) in baseImageHistograms.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        #print(f'Type {type(queryImageHistogram)}, shape {np.shape(queryImageHistogram)}')
        if not isinstance(queryImageHistogram, np.ndarray):
            query = cv2.UMat(np.array(queryImageHistogram, dtype=np.float32))
            histBase = cv2.UMat(np.array(hist, dtype=np.float32))
            distance = cv2.compareHist(query, histBase, comparisonMethod)
        else:
            distance = cv2.compareHist(queryImageHistogram, hist, comparisonMethod)
        # distance = chi2_distance(hist, queryImageHistogram)
        results[k] = distance
    return results

def chi2_distance(histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
		# return the chi-squared distance
		return d