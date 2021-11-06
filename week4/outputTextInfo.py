import cv2
import glob
import extractTextBox
import text_processing
import denoise_image
import background_processor
import pickle
from pathlib import Path
import numpy as np

#Open query image folder
query_image_folder = "/Users/brian/Desktop/Computer Vision/M1/Project/qsd1_w4"
filenames = [img for img in glob.glob(query_image_folder + "/*"+ ".jpg")]
filenames.sort()

#Load images to a list
images = []
for img in filenames:
    n = cv2.imread(img)
    images.append(n)

masks = []
TextBoxPickle = []
start, end = [], []

for i, inputImage in enumerate(images):
    print('Processing image: ', filenames[i])
    filename = filenames[i]
    queryImage = denoise_image.denoinseImage(inputImage)
    # cv2.imshow("queryImage", queryImage)

    backgroundMask, precision, recall, F1_measure = background_processor.backgroundRemoval(queryImage, filename)

    elems, start, end = background_processor.findElementsInMask(backgroundMask)

    texts = []

    if elems > 1:
        boxes = []
        for num in range(elems):
            auxMask = np.zeros(backgroundMask.shape, dtype="uint8")
            auxMask[start[num][0]:end[num][0],start[num][1]:end[num][1]] = 255
            # cv2.imshow("Background Mask", auxMask)

            res = cv2.bitwise_and(queryImage,queryImage,mask = auxMask)
            # cv2.imshow("Background Masked", res)

            textImage, textBoxMask, box = extractTextBox.getTextAlone(res)
            text = text_processing.imageToText(textImage)
            texts.append(text)
            # cv2.imshow("TextImage", textImage)
            # cv2.imshow("textBoxMask", textBoxMask)

            boxes.append(box)
            # cv2.waitKey(0)
        TextBoxPickle.append(boxes)
        print(texts)
    else:
        textImage, textMask, box = extractTextBox.getTextAlone(queryImage)
        text = text_processing.imageToText(textImage)
        print(text)
        texts.append(text)
        # cv2.imshow("TextImage", textImage)
        # cv2.imshow("textBoxMask", textMask)
        TextBoxPickle.append([box])

    with open((Path(filename).stem + '.txt'), 'w') as output:
        for text in texts:
            output.write(str(text) + '\n')

with open('text_boxes' + '.pkl', 'wb') as handle:
    pickle.dump(TextBoxPickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
