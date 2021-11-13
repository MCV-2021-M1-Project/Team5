import cv2
import os
from skimage.measure import ransac
from skimage.transform import AffineTransform
import numpy as np
from matplotlib import pyplot as plt

def loadAllImages(folderPath):
    
    ddbb_images = {}
    
    for img in filter(lambda el: el.find('.jpg') != -1, os.listdir(folderPath)):
        filename = folderPath + '/' + img
        image = cv2.imread(filename)

        # Store the image as RGB for later plot
        ddbb_images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return ddbb_images

def orderTuples(array, reverse = True):
    return sorted(array, reverse=reverse)

def keyPointMatching(kp1, des1, kp2, des2, img1, img2):
    if len(kp1) == 0 or len(kp2) == 0:
        return 0 

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
    # return len(good)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

    if (len(src_pts) > 4) and len(dst_pts) > 4:
        model, inliers = ransac(
                (src_pts, dst_pts),
                AffineTransform, min_samples=4,
                residual_threshold=8, max_trials=10
            )
        
        if inliers is not None:
            n_inliers = np.sum(inliers)
    
            inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
            inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
            placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
            # if len(inliers) > 30 :
            #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            #     image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None, -1)
            #     plt.title('After RANSAC')
            #     plt.imshow(image3)
            #     plt.show()
            src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
            dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
            return len(inliers)
        else:
            return 0
    else:
        return 0

def plotResults(ordered, images):
    fig,axes = plt.subplots(2, 5)
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(images[ordered[ind][1]])
        ax.set_title(ordered[ind][1])
        ax.axis('off')
    plt.tight_layout()
    plt.show()