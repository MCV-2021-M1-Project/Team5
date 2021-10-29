import glob
import string
import numpy as np
import pytesseract
import textdistance

# pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def imageToText(image):
    text = pytesseract.image_to_string(image)
    text = ' '.join(text.split())
    # print("Extracted Text: " + text)

    return text

def getImagesGtText(path):
    filenames = [img for img in glob.glob(path + "/*"+ ".txt")]
    filenames.sort()
    ddbb_texts = {}
    for ind, filename in enumerate(filenames):
        file1 = open(filename, 'r')
        Lines = file1.readlines()
        ddbb_texts[filename.replace('.txt', '.jpg')] = []
        if len(Lines) >= 1 and Lines[0][0] != '\n':
            for i, line in enumerate(Lines):
                ddbb_texts[filename.replace('.txt', '.jpg')].append(readTextFromFile(line))

    return ddbb_texts

def readTextFromFile(text):
    "".join(filter(lambda char: char in string.printable, text))

    painter_name = text.split(",", 1)[0]

    if painter_name.count("'") < 2:
        painter_name = (painter_name.split("\""))[1].split("\"")[0]
        ''.join(painter_name.split())
    else:
        painter_name = (painter_name.split("'"))[1].split("'")[0]
        ''.join(painter_name.split())
    # print("Ground Truth Text: " + painter_name)

    painting_name = text.split(",", 1)[1]
    if painting_name.count("'") < 2:
        painting_name = (painting_name.split("\""))[1].split("\"")[0]
        ''.join(painting_name.split())
    else:
        painting_name = (painting_name.split("'"))[1].split("'")[0]
        ''.join(painting_name.split())
    # print("Ground Truth Text1: " + painting_name)

    return painter_name, painting_name


def compareText(query_text, ddbb_text):
    """
    Compare the text of ddbb_texts with the one for queryImage and returns
    a dictionary of different methods

    :param queryHist: histogram of queryImage to search
    :param ddbb_histograms: dictionary with the histograms of the images where queryImage is going to be searched

    :return: Dictionary with all the distances for queryImage ordered
    """

    allResults = {}
    # Compute the distance to DDBB images with Hellinger distance metric
    if len(query_text) > 1:
        for idx, text in enumerate(query_text):
            results = getTextDistances(text, ddbb_text)
            # sort the results
            allResults[idx] = results
    else:
        results = getTextDistances(query_text, ddbb_text)
        # sort the results
        allResults[0] = results

    return allResults

def getTextDistances(query_text, ddbb_text):
    # loop over the index
    results = {}
    for (k, tuple) in ddbb_text.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        distance = 0
        if len(tuple) > 0:
            distance = textdistance.hamming.normalized_similarity(query_text, tuple[0][0])
            distance1 = textdistance.hamming.normalized_similarity(query_text, tuple[0][1])
            distance = max(distance, distance1)
        # distance = chi2_distance(hist, queryImageHistogram)
        results[k] = distance
    sorted_results = sorted([(v, k) for (k, v) in results.items()], reverse=True)
    return sorted_results