import os
import cv2
import glob
import argparse
from pathlib import Path
import pickle
import numpy as np
from matplotlib import pyplot as plt
import constants as C
from average_metrics import mapk

def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the task 1 script')
    parser.add_argument('-k', '--k_best', type=int, default=5, help='Number of images to retrieve')
    parser.add_argument('-p', '--path', default='./BBDD', type=str, help='Relative path to image folder')
    parser.add_argument('-c', '--color_space', default="Lab", type=str, help='Color space to use')
    parser.add_argument('-g', '--gt_results', type=str, default='gt_corresps.pkl', help='Relative path to the query ground truth results')
    parser.add_argument('-r', '--computed_results', type=str, default='result.pkl', help='Relative path to the computed results')
    parser.add_argument('-v', '--validation_metrics', type=bool, default=False, help='Set to true to extract the metrics')
    parser.add_argument('-q', '--query_image', type=str, help='Relative path to the query image')
    parser.add_argument('-f', '--query_image_folder', type=str, help='Relative path to the folder contining the query images')
    parser.add_argument('-m', '--mask', type=bool, default=False, help='Set True to remove background')
    parser.add_argument('-plt', '--plot_result', type=bool, default=False, help='Set to True to plot results')
    return parser.parse_args()

def getImagesAndHistograms(folderPath, colorSpace):
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

        results = getBestKCoincidences(method, ddbb_histograms, queryHist, k_best)

        # sort the results
        allResults[methodName] = sorted([(v, k) for (k, v) in results.items()], reverse=reverse)
    if mask_check:
        return allResults, precision, recall, F1_measure
    else:
        return allResults, 1, 1, 1

def getBestKCoincidences(comparisonMethod, baseImageHistograms, queryImageHistogram, k):
    # loop over the index
    results = {}
    for (k, hist) in baseImageHistograms.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        distance = cv2.compareHist(queryImageHistogram, hist, comparisonMethod)
        results[k] = distance
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

    # initialize the results figure
    fig, big_axes = plt.subplots(nrows=len(methodNames), ncols=1)
    fig.suptitle('')
    fig.tight_layout(h_pad=1.2)

    # set row names
    for row, big_ax in enumerate(big_axes, start=0):
        big_ax.set_title(methodNames[row], fontsize=10, y = 1.3)
        big_ax.axis("off")

    # plot each image in subplot
    for (j, (methodName, values)) in enumerate(results.items()):

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
    threshMinV, threshMaxV = 80, 255
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

def main():
    args = parse_args()

    if args.validation_metrics:
        with open(args.gt_results, 'rb') as reader:
            gtRes = pickle.load(reader)

        with open(args.computed_results, 'rb') as reader:
            computedRes = pickle.load(reader)
        
        resultScore = mapk(gtRes, computedRes, args.k_best)
        print(f'Average precision in {args.computed_results} for k = {args.k_best} is {resultScore}.')
        
    else:
        ddbb_images, ddbb_histograms = getImagesAndHistograms(args.path, args.color_space)

        # query either an image or a folder
        if args.query_image:
            queryImage = cv2.imread(args.query_image)
            filename = args.query_image
            comp = compareHistograms(queryImage, args.color_space, args.mask, args.k_best, ddbb_histograms, filename)
            allResults = comp[0]

            # plot K best coincidences
            if args.plot_result:
                # change the color space to RGB to plot the image later
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

            # Initialize result containers
            resultPickle = {}
            precisionList = []
            recallList = []
            F1List = []
            
            i = 0
            
            for queryImage in images:
                filename = filenames[i]
                i += 1
                comp = compareHistograms(queryImage, args.color_space, args.mask, args.k_best, ddbb_histograms, filename)
                allResults = comp[0]
                #Add the best k pictures to the array that is going to be exported as pickle
                for methodName, method in allResults.items():
                    bestPictures = []
                    if methodName not in resultPickle:
                        resultPickle[methodName] = []
                    for score, name in allResults[methodName][0:args.k_best]:
                        bestPictures.append(int(Path(name).stem.split('_')[1]))
                    resultPickle[methodName].append(bestPictures)
                
                precisionList.append(comp[1])
                recallList.append(comp[2])
                F1List.append(comp[3])
                
                if args.plot_result:
                    # change the color space to RGB to plot the image later
                    queryImageRGB = cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB)
                    plotResults(allResults, args.k_best, ddbb_images, queryImageRGB)
            
            # Mask evaluation results
            if args.mask and os.path.exists(args.gt_results):
                avgPrecision = sum(precisionList)/len(precisionList)
                print(f'Average precision of masks is {avgPrecision}.')
                avgRecall = sum(recallList)/len(recallList)
                print(f'Average recall of masks is {avgRecall}.')
                avgF1 = sum(F1List)/len(F1List)
                print(f'Average F1-measure of masks is {avgF1}.')

            #Result export
            gtRes = None
            if os.path.exists(args.gt_results):
                with open(args.gt_results, 'rb') as reader:
                    gtRes = pickle.load(reader)

            for name, res in resultPickle.items():
                if gtRes is not None:
                    resultScore = mapk(gtRes, res, args.k_best)
                    print(f'Average precision in {name} for k = {args.k_best} is {resultScore}.')
                with open(name + '_' + args.color_space + '.pkl', 'wb') as handle:
                    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()
