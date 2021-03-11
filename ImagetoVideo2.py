from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np
import cv2
from numpy import expand_dims
import tensorflow as tf
from matplotlib import pyplot
import natsort

pathout='videoAtoB(part2).avi'
fps=25
folderEvros="/data2/kazarian/PycharmProjects/TestingVol2ML/AtoB(part2)"
onlyfilesA = [f for f in os.listdir(folderEvros) if os.path.isfile(os.path.join(folderEvros, f))]
onlyfilesA=natsort.natsorted(onlyfilesA)
print("Working with {0} images".format(len(onlyfilesA)))
test_filesA = []

for _file in onlyfilesA:
    test_filesA.append(_file)
    print(_file)

print("Files in train_files: %d" % len(test_filesA))
image_width = 256
image_height = 256

channels = 3
nb_classes = 1

datasetA = np.ndarray(shape=(len(test_filesA), image_height, image_width, channels),
                      dtype=np.uint8)
i=0
for _file in test_filesA:
    img = load_img(folderEvros + "/" + _file, target_size=(image_width, image_height))  # this is a PIL image
    # Convert to Numpy Array
    x = img_to_array(img)
    # Normalize
    #x = (x - 127.5) / 127.5
    datasetA[i] = x
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)

print("All images to array!")
out=cv2.VideoWriter(pathout,cv2.VideoWriter_fourcc(*'MJPG'),fps,(256,256))
for y in range(len(datasetA)):
    out.write(datasetA[y])
out.release()