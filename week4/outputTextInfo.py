import cv2
import glob
import extractTextBox
import text_processing
import denoise_image
import background_processor
import textdistance
import pickle
from pathlib import Path
import numpy as np

def closingImage(gray, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return closing

def openingImage(gray, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    return opening

#Open query image folder
query_image_folder = "/Users/brian/Desktop/Computer Vision/M1/Project/qsd1_w4"
filenames = [img for img in glob.glob(query_image_folder + "/*"+ ".jpg")]
filenames.sort()

textnames = [text for text in glob.glob(query_image_folder + "/*" + ".txt")]
textnames.sort()

#Load images to a list
images = []
for img in filenames:
    n = cv2.imread(img)
    images.append(n)

masks = []
TextBoxPickle = []
start, end = [], []
distance_list = []

for i, inputImage in enumerate(images):
    print('Processing image: ', filenames[i])
    filename = filenames[i]

    queryImage = denoise_image.denoinseImage(inputImage)

    backgroundMask, precision, recall, F1_measure = background_processor.backgroundRemoval(queryImage, filename)

    elems, start, end = background_processor.findElementsInMask(backgroundMask)

    texts = []

    if elems > 1:
        boxes = []
        with open(textnames[i], encoding="latin-1") as file:
            lines = file.readlines()
        for num in range(elems):
            auxMask = np.zeros(backgroundMask.shape, dtype="uint8")
            auxMask[start[num][0]:end[num][0],start[num][1]:end[num][1]] = 255
            # cv2.imshow("Background Mask", auxMask)

            res = cv2.bitwise_and(queryImage,queryImage,mask = auxMask)
            # cv2.imshow("Background Masked", res)

            mask = contourText(queryImage)
            mask, x, y, w, h = extractTextBox.maskToRect(queryImage, mask)
            if w > 0 and h > 0:
                textImage = queryImage[y:y + h, x:x + w]
            else:
                textImage = queryImage

            # textImage, textBoxMask, box = extractTextBox.getTextAlone(res)
            extractedtext = text_processing.imageToText(textImage)
            texts.append(extractedtext)

            if lines[num]:
                painter_name, painting_name = text_processing.readTextFromFile(lines[num])
            else:
                painter_name = ""
                painting_name = ""

            distance = textdistance.hamming.normalized_similarity(extractedtext, painter_name)
            distance1 = textdistance.hamming.normalized_similarity(extractedtext, painting_name)
            distance = max(distance, distance1)
            print(distance)
            distance_list.append(distance)

            # cv2.imshow("TextImage", textImage)
            # cv2.imshow("textBoxMask", textBoxMask)

            # boxes.append(box)
            # cv2.waitKey(0)
        TextBoxPickle.append(boxes)
        print(texts)
    else:
        # textImage, textMask, box = extractTextBox.getTextAlone(queryImage)
        mask = contourText(queryImage)
        textImage = cv2.bitwise_and(queryImage, queryImage, mask=mask)

        extractedtext = text_processing.imageToText(textImage)


        with open(textnames[i], encoding="latin-1") as file:
            lines = file.readlines()
            painter_name, painting_name = text_processing.readTextFromFile(lines[0])

        distance = textdistance.hamming.normalized_similarity(extractedtext, painter_name)
        distance1 = textdistance.hamming.normalized_similarity(extractedtext, painting_name)
        distance = max(distance, distance1)

        print(distance)
        print(extractedtext)
        texts.append(extractedtext)
        # cv2.imshow("TextImage", textImage)
        # cv2.imshow("textBoxMask", textMask)
        # TextBoxPickle.append([box])

    with open((Path(filename).stem + '.txt'), 'w') as output:
        for text in texts:
            output.write(str(text) + '\n')

with open('text_boxes' + '.pkl', 'wb') as handle:
    pickle.dump(TextBoxPickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(TextBoxPickle)

print(np.average(distance_list))