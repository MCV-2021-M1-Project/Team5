import os
import cv2
import glob
import argparse
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from plot import plotResults
from timeit import default_timer as timer
from average_metrics import mapk
from matplotlib import pyplot as plt
from denoise_image import denoinseImage
from histogram_processing import loadColorHistograms, compareColorHistograms, getColorHistogramForQueryImage, loadAllImages
from texture_histograms import loadTextureHistograms, compareTextureHistograms, getTextureHistogramForQueryImage
from text_processing import getImagesGtText, compareText, imageToText
from extractTextBox import getTextAlone, EricText
from rotation import findAngle, rotate
from background_processor import backgroundFill, backgroundRemoval, findElementsInMask, crop_minAreaRect
from image_descriptors import loadImageDescriptors, getDescriptor, findBestMatches


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
    parser.add_argument('-w', '--weights', type=list, default=[0.2, 0.8, 0, 1], help='weights for combining descriptors')
    parser.add_argument('-kpd', '--keypoint_detection', type=str, default='ORB', help='Use keypoints to match images, the type of descriptor to use')
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
        ddbb_images = {}
        if args.plot_result:
            print('Loading all images for ploting...')
            ddbb_images = loadAllImages(args.path)
        
        #Loading or computing COLOR histograms for DDBB
        colorHistogramsFile = f'ddbb_color_histograms_{args.color_space}_segments{args.split}.pkl'
        ddbb_color_histograms = loadColorHistograms(colorHistogramsFile, args.path, args.color_space, args.split)
        

        #Loading or computing TEXTURE histograms for DDBB
        textureHistogramsFile = f'ddbb_texture_histograms_segments{args.split}.pkl'
        ddbb_texture_histograms = loadTextureHistograms(textureHistogramsFile, args.path, args.split)

        #Loading or computing Keypoint descriptors for DDBB
        descriptorFile = f'ddbb_{args.keypoint_detection}_descriptor.pkl'
        ddbb_descriptors = loadImageDescriptors(descriptorFile, args.path, args.keypoint_detection)
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
            resultKpMatchPickle = []
            textsPickle = []
            TextBoxPickle = []
            precisionList = []
            recallList = []
            F1List = []
            IoUList = []
            framePickle = []
            descriptor = getDescriptor(args.keypoint_detection)
            bestNumberOfMatches = []
            
            for i, inputImage in enumerate(images):
                print('Processing image: ', filenames[i])
                filename = filenames[i]
                queryImage = denoinseImage(inputImage)
                # Find mask if applicable
                backgroundMask = None
                precision, recall, F1_measure, IoU = -1, -1, -1, -1
                croppedImages = [queryImage]
                if args.mask:
                    maskFill = backgroundFill(queryImage)
                    angle = findAngle(maskFill)
                    
                    rotatedImage = rotate(queryImage, angle)
                    
                    rotatedMaskFill = backgroundFill(rotatedImage)
                    
                    # plt.imshow(cv2.cvtColor(rotatedImage, cv2.COLOR_BGR2RGB))
                    # plt.axis("off")
                    # plt.title("rotated")
                    # plt.show()
                    
                    # plt.imshow(cv2.cvtColor(rotatedMaskFill, cv2.COLOR_BGR2RGB))
                    # plt.axis("off")
                    # plt.title("rotated mask fill")
                    # plt.show()
                    
                    backgroundMask, precision, recall, F1_measure, IoU = backgroundRemoval(rotatedMaskFill, filename)

                    print("IoU: ",IoU)

                    # plt.imshow(cv2.cvtColor(backgroundMask, cv2.COLOR_GRAY2RGB))
                    # plt.axis("off")
                    # plt.title("rotatedMask")
                    # plt.show()

                    croppedImages = []
                    frames = []
                    contours, hierarchy = cv2.findContours(backgroundMask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


                    # #Converting to requested angle format
                    # if angle < 0:
                    #     angle = abs(angle)
                    # elif angle > 0:
                    #     angle = 180 - angle
                    
                    for cnt in contours:
                        rect = cv2.minAreaRect(cnt)

                        croppedImages.append(crop_minAreaRect(rotatedImage, rect))

                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        if rect[2] <= 0:
                            angle = 90 + abs(rect[2])
                        else:
                            angle = 180 - rect[2]
                            
                            
                        frame = [angle, box]
                        frames.append(frame)

                framePickle.append(frames)


                #Find text boxes and their masks if needed
                masks = []
                textBoxMasks = []
                textImages = []

                masks = []
                if args.extract_text_box:
                    for crop in croppedImages:
                        textImage, textBoxMask, box = getTextAlone(crop)
                        TextBoxPickle.append([box])
                        textBoxMasks.append(textBoxMask)
                        textImages.append(textImage)
                        backgroundMask = cv2.bitwise_not(textBoxMask)
                        masks.append(backgroundMask)

                #-------------------------------------------

                queryHistColor = []
                queryTextureHist = []
                allResultsDescriptors = {}
                if len(croppedImages) > 2:
                    print('Muchos crops')
                # plt.imshow(queryImage)
                for ind, img in enumerate(croppedImages):
                    # plt.imshow(img)
                    # plt.show()
                    queryHistColor.append(getColorHistogramForQueryImage(img, args.color_space, [masks[ind]], [], [], args.split)[0])
                    queryTextureHist.append(getTextureHistogramForQueryImage(img, [masks[ind]], [], [], args.split)[0])

                    imgFinal = img
                    if args.extract_text_box:
                        pixels = np.sum(masks[ind]) // 255
                        if pixels > 0.2 * (masks[ind].shape[0] * masks[ind].shape[1]):
                            masks[ind] = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
                            start = img.shape[0] // 4
                            start2 = img.shape[1] // 10
                            masks[ind][start:(img.shape[0]-start), :] = 255
                            imgFinal = cv2.bitwise_and(img,img,mask = (masks[ind]))
                        else:
                            imgFinal = cv2.bitwise_and(img,img,mask = cv2.bitwise_not(masks[ind]))
                    
                    resized = imgFinal
                    if imgFinal.shape[0] > 512 and imgFinal.shape[1] > 512:
                        resized = cv2.resize(imgFinal, (512, 512), interpolation = cv2.INTER_AREA)
                    queryKp, queryDescp = descriptor.detectAndCompute(resized, None)
                    allResultsDescriptors[ind] = findBestMatches(resized, queryKp, queryDescp, ddbb_descriptors, ddbb_images, args.keypoint_detection)
                    bestNumberOfMatches.append(allResultsDescriptors[ind][0][0])


                #Comparing COLOR histograms
                allResultsColor = compareColorHistograms(queryHistColor, ddbb_color_histograms)

                #Comparing TEXTURE histograms
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
                bestPicturesDescriptors, bestAuxDescriptors = [], []

                #Inistilise data frame
                all_result_df = pd.DataFrame(columns=["Image", "Color", "Texture", "Text", "Matches"])

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
                    
                    
                    # Add the best k pictures to the array that is going to be exported as pickle
                    if allResultsDescriptors[key][0][0] < 33:
                        bestAuxDescriptors = [-1]
                    else:
                        for score, name in allResultsDescriptors[key][0:args.k_best]:
                            bestAuxDescriptors.append(int(Path(name).stem.split('_')[1]))
                    bestPicturesDescriptors.append(bestAuxDescriptors)

                    for score, name in allResultsDescriptors[key]:
                        if score < 33:
                            all_result_df.loc[all_result_df["Image"] == name, "Matches"] = 0
                        else:
                            all_result_df.loc[all_result_df["Image"] == name, "Matches"] = score

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
                    all_result_df["Matches"]=(all_result_df["Matches"]-all_result_df["Matches"].min())/(all_result_df["Matches"].max()-all_result_df["Matches"].min())
                    # all_result_df["Text"]=(all_result_df["Text"]-all_result_df["Text"].min())/(all_result_df["Text"].max()-all_result_df["Text"].min())

                    weights = args.weights
                    all_result_df["Combined"] = 0
                    all_result_df["Combined"] = weights[0] * all_result_df["Color"] + weights[1] * all_result_df["Texture"] + weights[2] * all_result_df["Text"] + weights[3] * all_result_df["Matches"]

                    combinedResults = list(zip(all_result_df["Combined"].tolist(), all_result_df["Image"].tolist()))
                    combinedResults.sort(reverse=True)
                    if allResultsDescriptors[key][0][0] < 40:
                        bestAuxCombined = [-1]
                    else:
                        for scre, name in combinedResults[0:args.k_best]:
                            bestAuxCombined.append(int(Path(name).stem.split('_')[1]))
                    # print('Scores for combined: ',combinedResults[0:args.k_best])
                    bestPicturesCombined.append(bestAuxCombined)

                #pickel the k best results in lists of list
                resultPickleColor.append(bestPicturesColor)
                resultPickleTexture.append(bestPicturesTexture)
                resultKpMatchPickle.append(bestPicturesDescriptors)
                if args.extract_text_box:
                    resultPickleText.append(bestPicturesText)
                resultPickleCombined.append(bestPicturesCombined)
                #--------------------------

                #Expanding mask evaluation lists , recall, F1_measure
                precisionList.append(precision)
                recallList.append(recall)
                F1List.append(F1_measure)
                IoUList.append(IoU)

                if args.plot_result:
                    # change the color space to RGB to plot the image later
                    queryImageRGB = cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB)
                    plotResults(allResultsColor, args.k_best, ddbb_images, queryImageRGB)

            print('Best number of matches for each paint: ', bestNumberOfMatches)
            # Mask evaluation results
            if args.mask and os.path.exists(args.gt_results):
                avgPrecision = sum(precisionList)/len(precisionList)
                print(f'Average precision of masks is {avgPrecision}.')
                avgRecall = sum(recallList)/len(recallList)
                print(f'Average recall of masks is {avgRecall}.')
                avgF1 = sum(F1List)/len(F1List)
                print(f'Average F1-measure of masks is {avgF1}.')
                avgIoU = sum(IoUList) / len(IoUList)
                print(f'Average IoU of masks is {avgIoU}.')

            #Result export (Color)
            gtRes = None
            if os.path.exists(args.gt_results):
                with open(args.gt_results, 'rb') as reader:
                    gtRes = pickle.load(reader)

            if gtRes is not None:
                #COLOR
                flattened = [np.array(sublist).flatten() for sublist in resultPickleColor]
                resultScore = mapk(gtRes, flattened, args.k_best)
                print(f'Color average precision in Hellinger for k = {args.k_best} is {resultScore:.4f}.')
                #TEXTURE
                flattened = [np.array(sublist).flatten() for sublist in resultPickleTexture]
                resultScore = mapk(gtRes, flattened, args.k_best)
                print(f'Texture average precision in Hellinger for k = {args.k_best} is {resultScore:.4f}.')
                #TEXT
                if args.extract_text_box:
                    flattened = [np.array(sublist).flatten() for sublist in resultPickleText]
                    resultScore = mapk(gtRes, flattened, args.k_best)
                    print(f'Text average precision in Hellinger for k = {args.k_best} is {resultScore:.4f}.')
                    with open("results.txt", 'w') as output:
                        for row in textsPickle:
                            output.write(str(row) + '\n')
                    with open('text_boxes' + '.pkl', 'wb') as handle:
                        pickle.dump(TextBoxPickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #Descriptors
                flattened = [np.array(sublist).flatten() for sublist in resultKpMatchPickle]
                resultScore = mapk(gtRes, flattened, args.k_best)
                print(f'KeyPoint matching average precision for k = {args.k_best} is {resultScore:.4f}.')
                #Combined
                flattened = [np.array(sublist).flatten() for sublist in resultPickleCombined]
                resultScore = mapk(gtRes, flattened, args.k_best)
                print(f'Combined average precision for k = {args.k_best} is {resultScore}.')
            with open('Hellinger_' + args.color_space + '_segments' + str(args.split) + '.pkl', 'wb') as handle:
                pickle.dump(resultPickleCombined, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('frames' + '.pkl', 'wb') as handle:
                pickle.dump(framePickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #--------------------------------------

if __name__ == "__main__":
    main()