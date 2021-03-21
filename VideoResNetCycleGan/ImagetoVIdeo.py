from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
import cv2
from numpy import expand_dims
import tensorflow as tf
from matplotlib import pyplot
import natsort
import keras.backend as K
import gc

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

folderEvros="/data2/kazarian/Evros2"
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
                      dtype=np.float32)
i = 0
for _file in test_filesA:
    img = load_img(folderEvros + "/" + _file, target_size=(image_width, image_height))  # this is a PIL image
    # Convert to Numpy Array
    x = img_to_array(img)
    # Normalize
    x = (x - 127.5) / 127.5
    datasetA[i] = x
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)

print("All images to array!")
cust = {'InstanceNormalization': InstanceNormalization}

model_AtoB=load_model("generatorX_236050.h5", cust)
currentFrame = 0
for i in range(len(datasetA)):
    print(i)
    X= datasetA[i]
    X = expand_dims(X, 0)
    print(X.shape)
    B_generated = model_AtoB(tf.convert_to_tensor(X))
    proto = tf.make_tensor_proto(B_generated)
    B_generated=tf.make_ndarray(proto)
    K.clear_session()
    gc.collect()
    B_generated = (B_generated + 1) / 2.0
    #pyplot.imshow(B_generated[0])
    #pyplot.show()
    name='AtoB(part2)/' + str(currentFrame) + '.jpg'
    print('Creating...' + name)
    pyplot.imsave(name, B_generated[0])
    currentFrame+=1

