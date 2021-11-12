import cv2
import math
import statistics
from background_processor import backgroundFill, cleanerV, cleanerH, intersect_matrices
import numpy as np
from matplotlib import pyplot as plt

def findAngle(maskFill):     
    contours = cv2.Canny(maskFill,20,400)
    
    plt.imshow(cv2.cvtColor(contours, cv2.COLOR_GRAY2RGB))
    plt.axis("off")
    plt.show()
    
    lines = cv2.HoughLinesP(contours, 1, np.pi/180, 50, minLineLength = 150, maxLineGap = 20)
    angles = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        # cv2.line(queryImage,(x1,y1),(x2,y2),(0,0,255),10)
        
        angle = math.atan2(y2-y1, x2-x1) * (180.0 / math.pi)
        if angle < 0:
            if angle < -45:
                angle += 90
        else:
            if angle > 45:
                angle -= 90
        angles.append(angle)
        
    print(statistics.median(angles))
    return statistics.median(angles)

    # plt.imshow(cv2.cvtColor(maskFill, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.show()
    
def rotate(queryImage, angle):
    (h, w) = queryImage.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1)
    img_rot = cv2.warpAffine(queryImage, M, (w, h))
    
    return img_rot
    