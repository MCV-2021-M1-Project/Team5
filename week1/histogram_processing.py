import os
import cv2
import constants as C
from average_metrics import mapk
from background_processor import backgroundRemoval


def getImagesAndHistograms(folderPath, colorSpace):
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

        # Compute the histogram with color space passed as argument
        hist = cv2.calcHist([aux], channels, mask, bins, colorRange)
        hist = cv2.normalize(hist, hist).flatten()
        ddbb_histograms[filename] = hist
        
    return ddbb_images, ddbb_histograms



def compareHistograms(queryImage, colorSpace, mask_check, k_best, ddbb_histograms, filename):
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
    
    # Equalizing Saturation and Lightness via HSV
    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2HSV)
    h, s, v, = cv2.split(queryImage)
    eqV = cv2.equalizeHist(v)
    queryImage = cv2.merge((h, s, eqV))
    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_HSV2BGR)
    
    # Change to the color space that is going to be used to compare histograms
    queryImageColorSpace = cv2.cvtColor(queryImage, C.OPENCV_COLOR_SPACES[colorSpace][0])
    channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES[colorSpace][1:]
    
    # Apply mask if applicable
    if mask_check:
        mask, precision, recall, F1_measure = backgroundRemoval(queryImage, filename)

    # Compute the histogram with color space passed as argument
    queryHist = cv2.calcHist([queryImageColorSpace], channels, mask, bins, colorRange)
    queryHist = cv2.normalize(queryHist, queryHist).flatten()

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
        distance = cv2.compareHist(queryImageHistogram, hist, comparisonMethod)
        results[k] = distance
    return results