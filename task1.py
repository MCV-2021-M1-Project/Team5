import os
import cv2
import glob
import argparse

import pickle
import numpy as np
from matplotlib import pyplot as plt
import constants as C

def parse_args():
    parser = argparse.ArgumentParser(description= 'Test to parse args')
    parser.add_argument('-p', '--path', required=True, type=str, help='Relative path to dataset folder')
    parser.add_argument('-q', '--query_image', type=str, help='Relative path to the query image')
    parser.add_argument('-f', '--query_image_folder', type=str, help='Relative path to the folder containing the query images')
    parser.add_argument('-k', '--k_best', type=int, default=5, help='Number of images to retrieve')
    parser.add_argument('-c', '--color_space', default="Lab", type=str, help='Color space to use')
    parser.add_argument('-plt', '--plot_result', type=bool, default=False, help='Set to True to plot results')
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

def compareHistograms(queryImage, colorSpace, k_best, ddbb_histograms):
    # Change to the color space that is going to be used to compare histograms
    queryImageColorSpace = cv2.cvtColor(queryImage, C.OPENCV_COLOR_SPACES[colorSpace][0])
    channels, mask, bins, colorRange = C.OPENCV_COLOR_SPACES[colorSpace][1:]

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

        results = getBestKCoincidences(method, ddbb_histograms, queryHist, k_best)

        # sort the results
        allResults[methodName] = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)

    return allResults

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

    # get method names to a list
    methodNames = []
    for methodName, values in results.items():
        methodNames.append(methodName)

    #initialize the results figure
    fig, big_axes = plt.subplots(nrows=len(methodNames), ncols=1)
    fig.suptitle('')
    fig.tight_layout(h_pad=1.2)

    # set row names
    for row, big_ax in enumerate(big_axes, start=0):
        big_ax.set_title(methodNames[row], fontsize=10, y = 1.3)
        big_ax.axis("off")

    # plot each image in subplot
    for (j, (methodName, values)) in enumerate (results.items()):

        bestKValues = values[0:kBest]

        # loop over the results
        for (i, (v, k)) in enumerate(bestKValues):
            # show the result
            ax = fig.add_subplot(len(methodNames), kBest, j * kBest + i + 1)
            ax.set_title("%s: %.2f" % (os.path.basename(k), v), fontsize = 5)
            plt.imshow(imagesDDBB[k])
            plt.axis("off")

    # show the OpenCV methods
    plt.show()

def main():
    args = parse_args()
    ddbb_images, ddbb_histograms = getImagesAndHistograms(args.path, args.color_space)

    # query either an image or a folder
    if args.query_image:
        queryImage = cv2.imread(args.query_image)
        allResults = compareHistograms(queryImage, args.color_space, args.k_best, ddbb_histograms)

        # plot K best coincidences
        if args.plot_result:
            # chnage the color space to RGB to plot the image later
            queryImageRGB = cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB)
            plotResults(allResults, args.k_best, ddbb_images, queryImageRGB)

    elif args.query_image_folder:
        # Sort query images in alphabetical order
        filenames = [img for img in glob.glob(args.query_image_folder + "/*"+ ".jpg")]
        filenames.sort()

        # Load images to a list
        images = []
        for img in filenames:
            n = cv2.imread(img)
            images.append(n)

        for queryImage in images:
            allResults = compareHistograms(queryImage, args.color_space, args.k_best, ddbb_histograms)
            if args.plot_result:
                # chnage the color space to RGB to plot the image later
                queryImageRGB = cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB)
                plotResults(allResults, args.k_best, ddbb_images, queryImageRGB)
    else:
        print("No query")

if __name__ == "__main__":
    main()
