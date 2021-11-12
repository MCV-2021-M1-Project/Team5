import cv2
from matplotlib import pyplot as plt
import glob
import utils
import numpy as np


query_image_folder = "../../datasets/BBDD"

images = utils.loadAllImages(query_image_folder)

noises = []

for name, img in images.items():
    imgNoise = 0
    sectionsMask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    sH = img.shape[0] // 4
    sW = img.shape[1] // 4
    hStart, wStart = 0, 0
    for row in range(4):
        for column in range(4):
            sectionsMask[(hStart+sH*row):(sH*row + hStart + sH),(wStart+sW*column):(sW*column + wStart + sW)] = 255
            auxMasked = cv2.bitwise_and(img, img, mask=sectionsMask)   
            imgNoise += cv2.Laplacian(img, cv2.CV_64F).var()
            sectionsMask[:,:] = 0
    noises.append((imgNoise / 16, name))

ordered = utils.orderTuples(noises, True)

utils.plotResults(ordered, images)

