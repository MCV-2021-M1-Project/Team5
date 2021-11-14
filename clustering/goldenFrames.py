from matplotlib import pyplot as plt
import utils
import cv2
import numpy as np

ddbb_imgs = utils.loadAllImages('../../datasets/BBDD')
ddbb_color_histograms = {}
ddbb_color_histograms_frame = {}



start = 30
start2 = 30

score = []
for name, img in ddbb_imgs.items(): 
    mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    mask[start:(img.shape[0]-start), start2:(img.shape[1]-start2)] = 255
    mask = cv2.bitwise_not(mask)
    img_masked = cv2.bitwise_and(img,img, mask=mask)
    
    img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    # plt.imshow(img)
    # plt.axis("off")
    # plt.title("HSV")
    # plt.show()

    imgSegment = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    # plt.imshow(cv2.cvtColor(imgSegment, cv2.COLOR_HSV2RGB))
    # plt.axis("off")
    # plt.title("Masked")
    # plt.show()
    
    histH = cv2.calcHist([img_hsv], [0], mask, [180], [0, 179])
    histH = cv2.normalize(histH, histH).flatten()
    
    histS = cv2.calcHist([img_hsv], [1], mask, [100], [0, 255])
    histS = cv2.normalize(histS, histS).flatten()
    
    histV = cv2.calcHist([img_hsv], [2], mask, [100], [0, 255])
    histV = cv2.normalize(histV, histV).flatten()

    # utils.plotHistogram(histH,180)
    # utils.plotHistogram(histS)
    # utils.plotHistogram(histV)

    sumH = sum(histH[0:25])
    sumS = sum(histS[35:70])
    sumV = sum(histV[55:95])
    if sumH > 0.9 and sumS > 0.9 and sumV > 0.9:
        score.append(((sumH+sumS+sumV),name))
    else:
        score.append((0,name))

# Find wooden
ordered = utils.orderTuples(score, True)
print(ordered[0:5])
utils.plotResults(ordered, ddbb_imgs)