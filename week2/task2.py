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
from histogram_processing import getImagesAndHistograms, compareHistograms, getDistances, loadAllImages

def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the task 1 script')
    parser.add_argument('-k', '--k_best', type=int, default=5, help='Number of images to retrieve')
    parser.add_argument('-s', '--split', type=int, default=1, help='Before computing the histograms the image will be splited in SxS patches')
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


def main():
    args = parse_args()

    if args.validation_metrics:
        #read ground truth result (format [[r1],[r2]...])
        with open(args.gt_results, 'rb') as reader:
            gtRes = pickle.load(reader)

        #read ground result to compare (format [[r1],[r2]...])
        with open(args.computed_results, 'rb') as reader:
            computedRes = pickle.load(reader)
        
        resultScore = mapk(gtRes, computedRes, args.k_best)
        print(f'Average precision in {args.computed_results} for k = {args.k_best} is {resultScore}.')
        
    else:
        histogramsFile = f'ddbb_histograms_{args.color_space}_segments{args.split}.pkl'
        if os.path.exists(histogramsFile):
            #Load histograms for DB, they are always the same for a space color and split level
            with open(histogramsFile, 'rb') as reader:
                print('Load existing histograms...')
                ddbb_histograms = pickle.load(reader)
            ddbb_images = loadAllImages(args.path)
        else:
            ddbb_images, ddbb_histograms = getImagesAndHistograms(args.path, args.color_space, args.split)
            #Save histograms for next time
            with open(histogramsFile, 'wb') as handle:
                pickle.dump(ddbb_histograms, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # query either an image or a folder
        if args.query_image:
            queryImage = cv2.imread(args.query_image)
            filename = args.query_image
            comp = compareHistograms(queryImage, args.color_space, args.mask, args.k_best, ddbb_histograms, filename, args.split)
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
                comp = compareHistograms(queryImage, args.color_space, args.mask, args.k_best, ddbb_histograms, filename, args.split)
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
                # with open(name + '_' + args.color_space + '.pkl', 'wb') as handle:
                #     pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    main()
