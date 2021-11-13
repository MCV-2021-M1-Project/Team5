import os
import pickle
import utils
import cv2

ddbb_descriptors = {}
if os.path.exists('ddbb_ORB_descriptor.pkl'):
    #Load histograms for DB, they are always the same for a space color and split level
    with open('ddbb_ORB_descriptor.pkl', 'rb') as reader:
        print('Load existing descriptors...')
        ddbb_descriptors = pickle.load(reader)
        print('Done loading descriptors.')
else:
    exit(-1)

ddbb_imgs = utils.loadAllImages('../../datasets/BBDD')
queryImg = cv2.imread('shadow.png')

descriptor = cv2.ORB_create()
queryKp, queryDescp = descriptor.detectAndCompute(queryImg, None)
results = []
for name, dbDescp in ddbb_descriptors.items():
    dbKeypoint = []
    for kp in dbDescp[0]:
        dbKeypoint.append(cv2.KeyPoint(kp[0][0], kp[0][1], kp[1], kp[2], kp[3], kp[4], kp[5]))
    res = utils.keyPointMatching(queryKp, queryDescp, dbKeypoint, dbDescp[1], queryImg, ddbb_imgs[name])
    results.append((res, name))

ordered = utils.orderTuples(results, True)
print(ordered[0:5])
utils.plotResults(ordered, ddbb_imgs)