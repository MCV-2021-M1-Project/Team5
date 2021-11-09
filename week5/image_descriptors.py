import cv2
from matplotlib import pyplot as plt
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
from skimage.feature import match_descriptors
import numpy as np
import os
from tqdm import tqdm


def getDescriptor(descriptor):
    if descriptor == 'SIFT':
        return cv2.SIFT_create(500, sigma=1)
    elif descriptor == 'ORB':
        return cv2.ORB_create()
    elif descriptor == 'AKAZE':
        return cv2.AKAZE_create()
    else:
        return cv2.SIFT_create(500, sigma=1)


def getMatcher(descriptor):
    if descriptor == 'SIFT':
        return cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    elif descriptor == 'ORB':
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    elif descriptor == 'AKAZE':
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        return cv2.BFMatcher()


def getImagesDescriptors(folderPath, descriptorType):
    ddbb_descriptors = {}
    
    for img in filter(lambda el: el.find('.jpg') != -1, os.listdir(folderPath)):
        filename = folderPath + '/' + img
        image = cv2.imread(filename)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        descriptor = getDescriptor(descriptorType)
        resized = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
        kp1, des1 = descriptor.detectAndCompute(resized, None)
        kpTemp = [(point.pt, point.size, point.angle, point.response, point.octave, point.class_id) for point in kp1]

        ddbb_descriptors[filename] = (kpTemp, des1)
        
    return ddbb_descriptors


def keyPointMatching(img1, img2, kp1, des1, kp2, des2, descriptorType):
    if len(kp1) == 0 or len(kp2) == 0:
        return 0 

    matcher = getMatcher(descriptorType)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

    if (len(src_pts) > 4) and len(dst_pts) > 4:
        model, inliers = ransac(
                (src_pts, dst_pts),
                AffineTransform, min_samples=4,
                residual_threshold=4, max_trials=100, stop_sample_num=(int(len(src_pts) * 0.5))
            )
        
        if inliers is not None:
            # n_inliers = np.sum(inliers)
    
            # inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
            # inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
            # placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
            # if len(inliers) < 40 and len(inliers) > 20:
            #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            #     image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None, -1)
            #     plt.title('After RANSAC')
            #     plt.imshow(image3)
            #     plt.show()
            # src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
            # dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
            return len(inliers)
        else:
            return 0
    else:
        return 0

def findBestMatches(queryImg, queryKp, queryDescp, ddbb_descriptors, ddbb_images, descriptorType):
    bestMatches = {}

    for name, dbDescp in tqdm(ddbb_descriptors.items()):
        dbKeypoint = []
        for kp in dbDescp[0]:
            dbKeypoint.append(cv2.KeyPoint(kp[0][0], kp[0][1], kp[1], kp[2], kp[3], kp[4], kp[5]))
        resized = cv2.resize(ddbb_images[name], (512, 512), interpolation = cv2.INTER_AREA)
        matches = keyPointMatching(queryImg, resized, queryKp, queryDescp, dbKeypoint, dbDescp[1], descriptorType)
        bestMatches[name] = matches
        # if len(src_pts) > 24:
            # plt.title(name)
            # plt.imshow(ddbb_images[name])
            # plt.show()
            # print(f'Number of matches found for {name} is {matches}')
    result = sorted([(v, k) for (k, v) in bestMatches.items()], reverse=True)
    return result
