import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from retinaface import RetinaFace

path = "/Users/brian/Desktop/Computer Vision/M1/Project/BBDD"

retina_model = RetinaFace.build_model()

filenames = [img for img in glob.glob(path + "/*" + ".jpg")]
filenames.sort()

# Load images to a list
images = []
for img in filenames:
    # Pass all images through DeepFace
    faces = RetinaFace.extract_faces(filenames, model=retina_model)

    if len(faces) > 0:
        im = cv2.imread(img)
        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pixels = np.array(image)
        images.append(pixels)

fig,axes = plt.subplots(2, 3)
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(images[ind])
    ax.axis('off')
plt.tight_layout()
plt.show()