import glob
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import preprocess_input

path = "../../datasets/BBDD"
# resnet_weights_path = 'clustering/resnet50_coco_best_v2.1.0.h5'

my_new_model = keras.Sequential()
my_new_model.add(tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False
resnet_feature_list = []

filenames = [img for img in glob.glob(path + "/*" + ".jpg")]
filenames.sort()

# Load images to a list
images = []
for img in filenames:
    im = cv2.imread(img)
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pixels = np.array(image)
    images.append(pixels)
    im = cv2.resize(im, (224, 224))
    img = preprocess_input(np.expand_dims(im.copy(), axis=0))
    resnet_feature = my_new_model.predict(img)
    resnet_feature_np = np.array(resnet_feature)
    resnet_feature_list.append(resnet_feature_np.flatten())

array = np.array(resnet_feature_list)

kmeans = KMeans(n_clusters=30, random_state=5, n_init=20).fit(array)
print(kmeans.labels_)

c0 = []
c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
c7 = []
c8 = []
c9 = []
c10 = []
c11 = []
c12 = []
c13 = []
c14 = []
c15 = []
c16 = []
for i, k in enumerate(kmeans.labels_):
    if k == 0:
        c0.append(images[i])
    elif k == 1:
        c1.append(images[i])
    elif k == 2:
        c2.append(images[i])
    elif k == 3:
        c3.append(images[i])
    elif k == 4:
        c4.append(images[i])
    elif k == 5:
        c5.append(images[i])
    elif k == 6:
        c6.append(images[i])
    elif k == 7:
        c7.append(images[i])
    elif k == 8:
        c8.append(images[i])
    elif k == 9:
        c9.append(images[i])
    elif k == 10:
        c10.append(images[i])
    elif k == 11:
        c11.append(images[i])
    elif k == 12:
        c12.append(images[i])
    elif k == 13:
        c13.append(images[i])
    elif k == 14:
        c14.append(images[i])
    elif k == 15:
        c15.append(images[i])
    elif k == 16:
        c16.append(images[i])


if len(c0) > 1:
    fig,axes = plt.subplots(1, min(len(c0),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c0[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c1) > 1:
    fig,axes = plt.subplots(1, min(len(c1),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c1[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c2) > 1:
    fig,axes = plt.subplots(1, min(len(c2),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c2[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c3) > 1:
    fig,axes = plt.subplots(1, min(len(c3),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c3[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c4) > 1:
    fig,axes = plt.subplots(1, min(len(c4),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c4[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c5) > 1:
    fig,axes = plt.subplots(1, min(len(c5),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c5[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c6) > 1:
    fig,axes = plt.subplots(1, min(len(c6),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c6[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c7) > 1:
    fig,axes = plt.subplots(1, min(len(c7),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c7[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c8) > 1:
    fig,axes = plt.subplots(1, min(len(c8),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c8[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c9) > 1:
    fig,axes = plt.subplots(1, min(len(c9),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c9[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
if len(c11) > 1:
    fig,axes = plt.subplots(1, min(len(c11),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c11[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
if len(c12) > 1:
    fig,axes = plt.subplots(1, min(len(c12),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c12[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c13) > 1:
    fig,axes = plt.subplots(1, min(len(c13),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c13[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c14) > 1:
    fig,axes = plt.subplots(1, min(len(c14),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c14[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c15) > 1:
    fig,axes = plt.subplots(1, min(len(c15),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c15[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if len(c16) > 1:
    fig,axes = plt.subplots(1, min(len(c16),10))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(c16[ind])
        ax.axis('off')
    plt.tight_layout()
    plt.show()