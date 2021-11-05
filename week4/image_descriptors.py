import cv2
from matplotlib import pyplot as plt
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import numpy as np
import os


def getDescriptor(descriptor):
    if descriptor == 'SIFT':
        return cv2.xfeatures2d.SIFT_create(300)
    else:
        return cv2.xfeatures2d.SIFT_create(300)


def getImagesDescriptors(folderPath, descriptorType):
    ddbb_descriptors = {}
    
    for img in filter(lambda el: el.find('.jpg') != -1, os.listdir(folderPath)):
        filename = folderPath + '/' + img
        image = cv2.imread(filename)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        descriptor = getDescriptor(descriptorType)

        kp1, des1 = descriptor.detectAndCompute(image, None)
        kpTemp = [(point.pt, point.size, point.angle, point.response, point.octave, point.class_id) for point in kp1]

        ddbb_descriptors[filename] = (kpTemp, des1)
        
    return ddbb_descriptors


def keyPointMatching(img1, img2, kp1, des1, kp2, des2):
    # Input : image1 and image2 in opencv format
    # Output : corresponding keypoints for source and target images
    # Output Format : Numpy matrix of shape: [No. of Correspondences X 2] 

    # surf = cv2.xfeatures2d.SURF_create(100)
    # surf = cv2.xfeatures2d.SIFT_create()

    # kp1, des1 = descriptor.detectAndCompute(img1, None)
    # kp2, des2 = surf.detectAndCompute(img2, None)
    
    if len(kp1) == 0 or len(kp2) == 0:
        return [], [] 
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    des1_32 = np.float32(des1)
    des2_32 = np.float32(des2)
    matches = flann.knnMatch(des1_32, des2_32, k=2)
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Match descriptors.
    # matches = bf.knnMatch(des1,des2, k=2)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # Lowe's Ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

    # Ransac
    if (len(src_pts) > 4) and len(dst_pts) > 4:
        model, inliers = ransac(
                (src_pts, dst_pts),
                AffineTransform, min_samples=4,
                residual_threshold=8, max_trials=10000
            )
        
        if inliers is not None:
            n_inliers = np.sum(inliers)
    
            inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
            inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
            placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None, -1)
        
            plt.imshow(image3)
            plt.show()
            src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
            dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
            return src_pts, dst_pts
        else:
            return [], []
    
    else:
        return [], []

def findBestMatches(queryImg, queryKp, queryDescp, ddbb_descriptors, ddbb_images):

    for name, dbDescp in ddbb_descriptors.items():
        dbKeypoint = []
        for kp in dbDescp[0]:
            dbKeypoint.append(cv2.KeyPoint(kp[0][0], kp[0][1], kp[1], kp[2], kp[3], kp[4], kp[5]))
        src_pts, dst_pts = keyPointMatching(queryImg, ddbb_images[name], queryKp, queryDescp, dbKeypoint, dbDescp[1])
        if len(src_pts) > 2:
            print(f'Number of matches found for {name} is {len(src_pts)}')
