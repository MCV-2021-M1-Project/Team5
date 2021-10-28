import os
import cv2
import glob
import argparse
from pathlib import Path
import pickle
import numpy as np
from plot import plotResults
import constants as C
from average_metrics import mapk
from matplotlib import pyplot as plt
from denoise_image import denoinseImage
from histogram_processing import getColorHistograms, compareColorHistograms, getColorHistogramForQueryImage, getDistances, loadAllImages
from texture_histograms import getTextureHistograms, compareTextureHistograms, getTextureHistogramForQueryImage
from text_processing import getImagesGtText, compareText, imageToText
from extractTextBox import getTextAlone


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
    parser.add_argument('-d', '--denoise', type=bool, default=False, help='Denoise query image before processing it')
    return parser.parse_args()

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
            queryColorHist, _,_,_ = getColorHistogramForQueryImage(queryImage, args.color_space, args.mask, filename, args.split, args.extract_text_box)
            allResultsColor = compareColorHistograms(queryColorHist, ddbb_color_histograms)
            
            #Plot K best coincidences [B R O K E N] <------------

            if args.plot_result:
                #Change the color space to RGB to plot the image later
                queryImageRGB = cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB)
                plotResults(allResultsColor, args.k_best, ddbb_images, queryImageRGB)
            
            #Compare TEXTURE histograms
            queryTextureHist, _,_,_ = getTextureHistogramForQueryImage(queryImage, args.color_space, args.mask, filename, args.split, args.extract_text_box)
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
            precisionList = []
            recallList = []
            F1List = []
            
            
            for i, queryImage in enumerate(images):
                print('Processing image: ', filenames[i])
                filename = filenames[i]

                #Comparing COLOR histograms
                componentsColor = getColorHistogramForQueryImage(queryImage, args.color_space, args.mask, filename, args.split, args.extract_text_box)
                allResultsColor = compareColorHistograms(componentsColor[0], ddbb_color_histograms)

                #Add the best k pictures to the array that is going to be exported as pickle
                bestPictures = []
                bestAux = []
                for key, results in allResultsColor.items():
                    for score, name in results[0:args.k_best]:
                        bestAux.append(int(Path(name).stem.split('_')[1]))
                    bestPictures.append(bestAux)
                resultPickleColor.append(bestPictures)
                #--------------------------
                
                #Comparing TEXTURE histograms
                componentsTexture = getTextureHistogramForQueryImage(queryImage, args.color_space, args.mask, filename, args.split, args.extract_text_box)
                allResultsTexture = compareTextureHistograms(componentsTexture[0], ddbb_texture_histograms)
                
                # img_cropped = getTextAlone(components[0])
                # query_text = imageToText(img_cropped)
                # textResults = compareText(query_text, ddbb_text)


                #Add the best k pictures to the array that is going to be exported as pickle
                bestPictures = []
                bestAux = []
                for key, results in allResultsTexture.items():
                    for score, name in results[0:args.k_best]:
                        bestAux.append(int(Path(name).stem.split('_')[1]))
                    bestPictures.append(bestAux)
                resultPickleTexture.append(bestPictures)
                #--------------------------
                
                #Expanding mask evaluation lists
                precisionList.append(componentsColor[1])
                recallList.append(componentsColor[2])
                F1List.append(componentsColor[3])
                
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
            with open('Hellinger_' + args.color_space + '_segments' + str(args.split) + '.pkl', 'wb') as handle:
                pickle.dump(resultPickleColor, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #--------------------------------------

if __name__ == "__main__":
    main()
