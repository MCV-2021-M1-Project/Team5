import os
import cv2
import glob
import argparse
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from plot import plotResults
import constants as C
from average_metrics import mapk, normalizeTupleVector
from matplotlib import pyplot as plt
from denoise_image import denoinseImage
from histogram_processing import getColorHistograms, compareColorHistograms, getColorHistogramForQueryImage, loadAllImages
from texture_histograms import getTextureHistograms, compareTextureHistograms, getTextureHistogramForQueryImage
from text_processing import getImagesGtText, compareText, imageToText
from extractTextBox import getTextBoundingBoxAlone, getTextAlone
from background_processor import backgroundRemoval, findElementsInMask


def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the task 1 script')
    parser.add_argument('-k', '--k_best', type=int, default=5, help='Number of images to retrieve')
    parser.add_argument('-s', '--split', type=int, default=1, help='Before computing the histograms the image will be split into SxS patches')
    parser.add_argument('-p', '--path', default='./BBDD', type=str, help='Relative path to image folder')
    parser.add_argument('-c', '--color_space', default="Lab", type=str, help='Color space to use')
    parser.add_argument('-g', '--gt_results', type=str, default='gt_corresps.pkl', help='Relative path to the query ground truth results')
    parser.add_argument('-r', '--computed_results', type=str, default='result.pkl', help='Relative path to the computed results')
    parser.add_argument('-v', '--validation_metrics', type=bool, default=False, help='Set to true to extract the metrics')
    parser.add_argument('-q', '--query_image', type=str, help='Relative path to the query image')
    parser.add_argument('-f', '--query_image_folder', type=str, help='Relative path to the folder contining the query images')
    parser.add_argument('-m', '--mask', type=bool, default=False, help='Set True to remove background')
    parser.add_argument('-t', '--extract_text_box', type=bool, default=False, help='Set True to extract the text bounding box')
    parser.add_argument('-plt', '--plot_result', type=bool, default=False, help='Set to True to plot results')
    parser.add_argument('-w', '--weights', type=list, default=[0, 1, 0], help='weights for combining descriptors')
    return parser.parse_args()

def oneTake(x):
    return 1-x

def main():
    args = parse_args()

    if args.validation_metrics:
        #read ground truth result (format [[r1],[r2]...])
        with open(args.gt_results, 'rb') as reader:
            gtRes = pickle.load(reader)

        #Read ground result to compare (format [[r1],[r2]...])
        with open(args.computed_results, 'rb') as reader:
            computedRes = pickle.load(reader)
        
        resultScore = mapk(gtRes, computedRes, args.k_best)
        print(f'Average precision in {args.computed_results} for k = {args.k_best} is {resultScore}.')
        
    else:
        #---------PREPARING DDBB DATA----------
        #Loading DDBB images
        if args.plot_result:
            print('Loading all images for ploting...')
            ddbb_images = loadAllImages(args.path)
        
        #Loading or computing COLOR histograms for DDBB
        colorHistogramsFile = f'ddbb_color_histograms_{args.color_space}_segments{args.split}.pkl'
        if os.path.exists(colorHistogramsFile):
            #Load histograms for DB, they are always the same for a space color and split level
            with open(colorHistogramsFile, 'rb') as reader:
                print('Load existing color histograms...')
                ddbb_color_histograms = pickle.load(reader)
                print('Done loading color histograms.')
        else:
            ddbb_color_histograms = getColorHistograms(args.path, args.color_space, args.split)
            #Save histograms for next time
            with open(colorHistogramsFile, 'wb') as handle:
                pickle.dump(ddbb_color_histograms, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #Loading or computing TEXTURE histograms for DDBB
        textureHistogramsFile = f'ddbb_texture_histograms_segments{args.split}.pkl'
        if os.path.exists(textureHistogramsFile):
            #Load histograms for DB
            with open(textureHistogramsFile, 'rb') as reader:
                print('Load existing texture histograms...')
                ddbb_texture_histograms = pickle.load(reader)
                print('Done loading texture histograms.')
        else:
            ddbb_texture_histograms = getTextureHistograms(args.path, args.split)
            #Save histograms for next time
            with open(textureHistogramsFile, 'wb') as handle:
                pickle.dump(ddbb_texture_histograms, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #--------------------------------------

        #Read the text files for data base
        ddbb_text = getImagesGtText(args.path)
        
        #--PREPARING QUERY DATA AND COMPARING--
        #Query either an image or a folder
        if args.query_image:
            queryImage = cv2.imread(args.query_image)
            queryImageDenoised = denoinseImage(queryImage)
            filename = args.query_image

            #Compare COLOR histograms
            queryColorHist, _,_,_ = getColorHistogramForQueryImage(queryImageDenoised, args.color_space, args.mask, filename, args.split, args.extract_text_box)
            allResultsColor = compareColorHistograms(queryColorHist, ddbb_color_histograms)
            
            #Plot K best coincidences <------------

            if args.plot_result:
                #Change the color space to RGB to plot the image later
                queryImageRGB = cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB)
                plotResults(allResultsColor, args.k_best, ddbb_images, queryImageRGB)
            
            #Compare TEXTURE histograms
            queryTextureHist, _,_,_ = getTextureHistogramForQueryImage(queryImageDenoised, args.color_space, args.mask, filename, args.split, args.extract_text_box)
            allResultsTexture = compareTextureHistograms(queryTextureHist, ddbb_texture_histograms)
            
        elif args.query_image_folder:
            #Sort query images in alphabetical order
            filenames = [img for img in glob.glob(args.query_image_folder + "/*"+ ".jpg")]
            filenames.sort()

            #Load images to a list
            images = []
            for img in filenames:
                n = cv2.imread(img)
                images.append(n)

            #Initialize result containers
            resultPickleColor = []
            resultPickleTexture = []
            resultPickleText = []
            resultPickleCombined = []
            textsPickle = []
            TextBoxPickle = []
            precisionList = []
            recallList = []
            F1List = []
            
            
            for i, inputImage in enumerate(images):
                print('Processing image: ', filenames[i])
                filename = filenames[i]
                queryImage = denoinseImage(inputImage)
                # Find mask if applicable
                backgroundMask = None
                precision, recall, F1_measure = -1, -1, -1
                if args.mask:
                    backgroundMask, precision, recall, F1_measure = backgroundRemoval(queryImage, filename)
                    # backgroundMask = cv2.imread(filename.replace('jpg','png'), cv2.IMREAD_GRAYSCALE)

                #Find text boxes and their masks if needed
                masks = []
                textBoxMasks = []
                textImages = []
                start, end = [], []

                if backgroundMask is None:
                    if args.extract_text_box:
                        textImage, textBoxMask, box = getTextAlone(queryImage)
                        TextBoxPickle.append([box])
                        textBoxMasks.append(textBoxMask)
                        textImages.append(textImage)
                        backgroundMask = cv2.bitwise_not(textBoxMask)
                        # plt.imshow(cv2.cvtColor(backgroundMask, cv2.COLOR_BGR2RGB))
                        # plt.axis("off")
                        # plt.show()
                        masks.append(backgroundMask)
                else:
                    elems, start, end = findElementsInMask(backgroundMask)
                    if elems > 1:
                        boxes = []
                        for num in range(elems):
                            auxMask = np.zeros(backgroundMask.shape, dtype="uint8")
                            auxMask[start[num][0]:end[num][0],start[num][1]:end[num][1]] = 255
                            # cv2.imshow("Background Mask", auxMask)
                            # cv2.waitKey(0)
                            if args.extract_text_box:
                                res = cv2.bitwise_and(queryImage,queryImage,mask = auxMask)
                                textImage, textBoxMask, box = getTextAlone(res)
                                # cv2.imshow("TextImage", textImage)
                                # cv2.imshow("textBoxMask", textBoxMask)
                                boxes.append(box)
                                textBoxMasks.append(textBoxMask)
                                textImages.append(textImage)
                                auxMask = cv2.bitwise_and(auxMask,auxMask,mask = cv2.bitwise_not(textBoxMask))
                            # plt.imshow(cv2.cvtColor(auxMask, cv2.COLOR_BGR2RGB))
                            # plt.axis("off")
                            # plt.show()
                            masks.append(auxMask)
                        TextBoxPickle.append(boxes)
                    else:
                        if args.extract_text_box:
                            res = cv2.bitwise_and(queryImage,queryImage,mask = backgroundMask)
                            textImage, textMask, box = getTextAlone(res)
                            textImages.append(textImage)
                            TextBoxPickle.append([box])
                            auxMask = cv2.bitwise_and(backgroundMask,backgroundMask,mask = cv2.bitwise_not(textMask))
                            # plt.imshow(cv2.cvtColor(auxMask, cv2.COLOR_BGR2RGB))
                            # plt.axis("off")
                            # plt.show()
                            masks.append(auxMask)
                        else:
                            # plt.imshow(cv2.cvtColor(backgroundMask, cv2.COLOR_BGR2RGB))
                            # plt.axis("off")
                            # plt.show()
                            masks.append(backgroundMask)

                #-------------------------------------------

                #Comparing COLOR histograms
                queryHistColor = getColorHistogramForQueryImage(queryImage, args.color_space, masks, start, end, args.split)
                allResultsColor = compareColorHistograms(queryHistColor, ddbb_color_histograms)

                #Comparing TEXTURE histograms
                queryTextureHist = getTextureHistogramForQueryImage(queryImage, masks, start, end, args.split)
                allResultsTexture = compareTextureHistograms(queryTextureHist, ddbb_texture_histograms)

                # Comparing TEXT
                if args.extract_text_box:
                    queryTexts = []
                    for textImg in textImages:
                        text = imageToText(textImg)
                        if len(text) > 1:
                            text.replace('\n','')
                        textsPickle.append(text)
                        queryTexts.append(text)
                    with open((Path(filename).stem + '.txt'), 'w') as output:
                        for row in queryTexts:
                            output.write(str(row) + '\n')    
                    allResultsText = compareText(queryTexts, ddbb_text)

                #Add the best k pictures to the array that is going to be exported as pickle
                bestPicturesColor, bestAuxColor = [], []
                bestPicturesTexture, bestAuxTexture = [], []
                bestPicturesText, bestAuxText = [], []
                bestPicturesCombined, bestAuxCombined = [], []

                #Inistilise data frame
                all_result_df = pd.DataFrame(columns=["Image", "Color", "Texture", "Text"])

                for key, results in allResultsColor.items():
                    # Creates DataFrame
                    all_result_df = pd.DataFrame(data=allResultsColor[key], columns=["Color", "Image"])
                    columns_titles = ["Image", "Color"]
                    all_result_df = all_result_df.reindex(columns=columns_titles)
                    all_result_df["Texture"] = 0
                    all_result_df["Text"] = 0

                    # Add the best k pictures to the array that is going to be exported as pickle
                    for score, name in results[0:args.k_best]:
                        bestAuxColor.append(int(Path(name).stem.split('_')[1]))
                    bestPicturesColor.append(bestAuxColor)

                    # Add the best k pictures to the array that is going to be exported as pickle
                    for score, name in allResultsTexture[key][0:args.k_best]:
                        bestAuxTexture.append(int(Path(name).stem.split('_')[1]))
                    bestPicturesTexture.append(bestAuxTexture)

                    for score, name in allResultsTexture[key]:
                        all_result_df.loc[all_result_df["Image"] == name, "Texture"] = score

                    if args.extract_text_box and bool(allResultsText):
                        for score, name in allResultsText[key][0:args.k_best]:
                            bestAuxText.append(int(Path(name).stem.split('_')[1]))
                        bestPicturesText.append(bestAuxText)
                        # print('Scores for text:',allResultsText[key][0:args.k_best])

                        for score, name in allResultsText[key]:
                            all_result_df.loc[all_result_df["Image"] == name, "Text"] = score

                    # Revert the score
                    all_result_df["Color"] = all_result_df["Color"].map(oneTake)
                    all_result_df["Texture"] = all_result_df["Texture"].map(oneTake)

                    # Normalise the final score with min-max normalization
                    all_result_df["Color"]=(all_result_df["Color"]-all_result_df["Color"].min())/(all_result_df["Color"].max()-all_result_df["Color"].min())
                    all_result_df["Texture"]=(all_result_df["Texture"]-all_result_df["Texture"].min())/(all_result_df["Texture"].max()-all_result_df["Texture"].min())
                    # all_result_df["Text"]=(all_result_df["Text"]-all_result_df["Text"].min())/(all_result_df["Text"].max()-all_result_df["Text"].min())

                    weights = args.weights
                    all_result_df["Combined"] = 0
                    all_result_df["Combined"] = weights[0] * all_result_df["Color"] + weights[1] * all_result_df["Texture"] + weights[2] * all_result_df["Text"]

                    combinedResults = list(zip(all_result_df["Combined"].tolist(), all_result_df["Image"].tolist()))
                    combinedResults.sort(reverse=True)
                    for scre, name in combinedResults[0:args.k_best]:
                        bestAuxCombined.append(int(Path(name).stem.split('_')[1]))
                    # print('Scores for combined: ',combinedResults[0:args.k_best])
                    bestPicturesCombined.append(bestAuxCombined)

                #pickel the k best results in lists of list
                resultPickleColor.append(bestPicturesColor)
                resultPickleTexture.append(bestPicturesTexture)
                if args.extract_text_box:
                    resultPickleText.append(bestPicturesText)
                resultPickleCombined.append(bestPicturesCombined)
                #--------------------------

                #Expanding mask evaluation lists , recall, F1_measure
                precisionList.append(precision)
                recallList.append(recall)
                F1List.append(F1_measure)

                if args.plot_result:
                    # change the color space to RGB to plot the image later
                    queryImageRGB = cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB)
                    plotResults(allResultsColor, args.k_best, ddbb_images, queryImageRGB)

            # Mask evaluation results
            if args.mask and os.path.exists(args.gt_results):
                avgPrecision = sum(precisionList)/len(precisionList)
                print(f'Average precision of masks is {avgPrecision}.')
                avgRecall = sum(recallList)/len(recallList)
                print(f'Average recall of masks is {avgRecall}.')
                avgF1 = sum(F1List)/len(F1List)
                print(f'Average F1-measure of masks is {avgF1}.')

            #Result export (Color)
            gtRes = None
            if os.path.exists(args.gt_results):
                with open(args.gt_results, 'rb') as reader:
                    gtRes = pickle.load(reader)

            if gtRes is not None:
                #COLOR
                flattened = [np.array(sublist).flatten() for sublist in resultPickleColor]
                resultScore = mapk(gtRes, flattened, args.k_best)
                print(f'Color average precision in Hellinger for k = {args.k_best} is {resultScore}.')
                #TEXTURE
                flattened = [np.array(sublist).flatten() for sublist in resultPickleTexture]
                resultScore = mapk(gtRes, flattened, args.k_best)
                print(f'Texture average precision in Hellinger for k = {args.k_best} is {resultScore}.')
                #TEXT
                if args.extract_text_box:
                    flattened = [np.array(sublist).flatten() for sublist in resultPickleText]
                    resultScore = mapk(gtRes, flattened, args.k_best)
                    print(f'Text average precision in Hellinger for k = {args.k_best} is {resultScore}.')
                    with open("results.txt", 'w') as output:
                        for row in textsPickle:
                            output.write(str(row) + '\n')
                    print(TextBoxPickle)
                    with open('text_boxes' + '.pkl', 'wb') as handle:
                        pickle.dump(TextBoxPickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #Combined
                flattened = [np.array(sublist).flatten() for sublist in resultPickleCombined]
                resultScore = mapk(gtRes, flattened, args.k_best)
                print(f'Combined average precision for k = {args.k_best} is {resultScore}.')
            with open('Hellinger_' + args.color_space + '_segments' + str(args.split) + '.pkl', 'wb') as handle:
                pickle.dump(resultPickleCombined, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #--------------------------------------

if __name__ == "__main__":
    main()