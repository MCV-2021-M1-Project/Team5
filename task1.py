import os
import argparse

import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
import constants as C

def parse_args():
    parser = argparse.ArgumentParser(description= 'Test to parse args')
    parser.add_argument('-k', '--k_best', type=int, default=4, help='Number of images to retrieve')
    parser.add_argument('-p', '--path', required=True, type=str, help='Relative path to image folder')
    parser.add_argument('-c', '--color_space', default="Lab", type=str, help='Color space to use')
    parser.add_argument('-q', '--query_image', default="./BBDD/query.jpg", type=str, help='Relative path to the query image')
    return parser.parse_args()

def getImagesAndHistograms(folderPath, colorSpace):
    ddbb_images = {}
    ddbb_histograms = {}
    
    for img in filter(lambda el: el.find('.jpg') != -1, os.listdir(folderPath)):
        filename = folderPath + '/' + img
        image = cv2.imread(filename)
        aux = cv2.cvtColor(image, C.OPENCV_COLOR_SPACES[colorSpace][0])
        ddbb_images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Storage the image as RGB for later plot
        channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES[colorSpace][1:]
        #Compute the histogram with color space passed as argument
        hist = cv2.calcHist([aux], channels, mask, bins, colorRange)
        hist = cv2.normalize(hist, hist).flatten()
        ddbb_histograms[filename] = hist
        
    return ddbb_images, ddbb_histograms

def getBestKCoincidences(comparisonMethod, baseImageHistograms, queryImageHistogram, k):
    # loop over the index
    results = {}
    for (k, hist) in baseImageHistograms.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        d = cv2.compareHist(queryImageHistogram, hist, comparisonMethod)
        results[k] = d
    return results


def plotResults(results, kBest, imagesDDBB, queryImage):
    # show the query image
    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(queryImage)
    plt.axis("off")
    
    for methodName, values in results.items():
        
        bestKValues = values[0:kBest]
        
        # initialize the results figure
        fig = plt.figure("Results: %s" % (methodName))
        fig.suptitle(methodName, fontsize = 20)
        # loop over the results
        for (i, (v, k)) in enumerate(bestKValues):
            # show the result
            ax = fig.add_subplot(1, len(bestKValues), i + 1)
            ax.set_title("%s: %.2f" % (k, v))
            plt.imshow(imagesDDBB[k])
            plt.axis("off")
            # show the OpenCV methods
    plt.show()

def main():
    args = parse_args()
    ddbb_images, ddbb_histograms = getImagesAndHistograms(args.path, args.color_space)
    queryImage = cv2.imread(args.query_image)
    #Change to the color space that is going to be used to compare histograms
    queryImageColorSpace = cv2.cvtColor(queryImage, C.OPENCV_COLOR_SPACES[args.color_space][0])
    channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES[args.color_space][1:]
    #Compute the histogram with color space passed as argument
    queryHist = cv2.calcHist([queryImageColorSpace], channels, mask, bins, colorRange)
    queryHist = cv2.normalize(queryHist, queryHist).flatten()
    #chnage the color space to RGB to plot the image later
    queryImageRGB = cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB)
    
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
            
            
        results = getBestKCoincidences(method, ddbb_histograms, queryHist, args.k_best)
            
           # sort the results
        allResults[methodName] = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)
        # show the query image
        
    plotResults(allResults, args.k_best, ddbb_images, queryImageRGB)
            

if __name__ == "__main__":
    main()
