import glob
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import preprocess_input

path = "/Users/brian/Desktop/Computer Vision/M1/Project/BBDD"
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

kmeans = KMeans(n_clusters=10, random_state=0).fit(array)
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



fig,axes = plt.subplots(1, min(len(c0),5))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(c0[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()

fig,axes = plt.subplots(1, min(len(c1),5))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(c1[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()

fig,axes = plt.subplots(1, min(len(c2),5))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(c2[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()

fig,axes = plt.subplots(1, min(len(c3),5))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(c3[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()

fig,axes = plt.subplots(1, min(len(c4),5))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(c4[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()

fig,axes = plt.subplots(1, min(len(c5),5))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(c5[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()

fig,axes = plt.subplots(1, min(len(c6),5))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(c6[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()

fig,axes = plt.subplots(1, min(len(c7),5))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(c7[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()

fig,axes = plt.subplots(1, min(len(c8),5))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(c8[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()

fig,axes = plt.subplots(1, min(len(c9),5))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(c9[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()