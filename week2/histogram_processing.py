import os
import cv2
import numpy as np
import constants as C
from average_metrics import mapk
from background_processor import backgroundRemoval

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

        hist = None
        if sections <= 1:
        # Compute the histogram with color space passed as argument
            hist = cv2.calcHist([aux], channels, mask, bins, colorRange)
            hist = cv2.normalize(hist, hist).flatten()
        else:
            sectionsMask = np.zeros((aux.shape[0], aux.shape[1]), dtype="uint8")
            sH = aux.shape[0] // sections
            sW = aux.shape[1] // sections
            auxHists = []
            for row in range(sections):
                for column in range(sections):
                    sectionsMask[sH*row:(sH*row + sH),sW*column:(sW*column + sW)] = 255
                    auxhist = cv2.calcHist([aux], channels, sectionsMask, bins, colorRange)
                    auxHists.extend(cv2.normalize(auxhist, auxhist).flatten())
                    sectionsMask[:,:] = 0
            hist = auxHists

        ddbb_histograms[filename] = hist
        
    return ddbb_images, ddbb_histograms



def compareHistograms(queryImage, colorSpace, mask_check, k_best, ddbb_histograms, filename, sections = 1):
    """
    Compare the histograms of ddbb_histograms with the one for queryImage and returns
    a dictionary of diferent methods with the k_best most similar images

    :param queryImage: image to look for in ddbb_histograms
    :param colorSpace: color space to compute the histogramas of the query image
    :param mask_check: if true tried to remove the background from queryImage
    :param k_best: number of images to retrieve
    :param ddbb_histograms: dictionary with the histograms of the images where queryImage is going to be searched
    :param filename: if mask_check is true it's used to load the gt mask and compute the quality of the computed mask

    :return: 1- Dictionary with all the distances for queryImage ordered for different Methods (format: {'MethodName': [Distances...]})
             2- precsion of the mask computed if mask_check and filename has a png with the ground truth, -1 othewise
             3- recall of the mask computed if mask_check and filename has a png with the ground truth, -1 othewise
             4- f1-measure of the mask computed if mask_check and filename has a png with the ground truth, -1 othewise
    """ 
    # Denoising query image using Gaussian blur
    queryImage = cv2.GaussianBlur(queryImage,(3,3), 0)

    # Apply mask if applicable
    if mask_check:
        mask, precision, recall, F1_measure = backgroundRemoval(queryImage, filename)

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
    queryHist = None
    if sections <= 1:
        # Compute the histogram with color space passed as argument
            queryHist = cv2.calcHist([queryImageColorSpace], channels, mask, bins, colorRange)
            queryHist = cv2.normalize(queryHist, queryHist).flatten()
    else:
        sectionsMask = np.zeros((queryImageColorSpace.shape[0], queryImageColorSpace.shape[1]), dtype="uint8")
        sH = queryImageColorSpace.shape[0] // sections
        sW = queryImageColorSpace.shape[1] // sections
        auxHists = []
        print(f'For query image {filename} the shape is {queryImageColorSpace.shape} and its sections are {sH}, {sW}')
        for row in range(sections):
            for column in range(sections):
                # print(f'Segment {row},{column}: {sH*row}:{(sH*row + sH)} {sW*column}:{(sW*column + sW)}')
                sectionsMask[sH*row:(sH*row + sH),sW*column:(sW*column + sW)] = 255
                auxhist = cv2.calcHist([queryImageColorSpace], channels, sectionsMask, bins, colorRange)
                auxHists.extend(cv2.normalize(auxhist, auxhist).flatten())
                sectionsMask[:,:] = 0
        queryHist = auxHists


    allResults = {}

    # loop over the comparison methods
    for (methodName, method) in C.OPENCV_METHODS:
        # initialize the results dictionary and the sort
        # direction
        results = {}
        reverse = False

        # if we are using the correlation or intersection
        # method, then sort the results in reverse order
        if methodName in ("Correlation", "Intersection"):
            reverse = True

        results = getDistances(method, ddbb_histograms, queryHist)

        # sort the results
        allResults[methodName] = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
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
        query = cv2.UMat(np.array(queryImageHistogram, dtype=np.float32))
        histBase = cv2.UMat(np.array(hist, dtype=np.float32))
        distance = cv2.compareHist(query, histBase, comparisonMethod)
        # distance = chi2_distance(hist, queryImageHistogram)
        results[k] = distance
    return results

def chi2_distance(histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
		# return the chi-squared distance
		return d