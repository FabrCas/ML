import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

datadir = "Dataset"

def loadData():
    batch_size = 64
    input_shape = ()
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,\
        zoom_range=0.1,\
        rotation_range=10,\
        width_shift_range=0.1,\
        height_shift_range=0.1,\
        horizontal_flip=True,\
        vertical_flip=False,\
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        directory=datadir,
        target_size=(118, 224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        directory=datadir,  # same as training data
        target_size=(118, 224),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical',
        subset='validation')  # set as validation data

    n_samples = train_generator.n
    n_classes = train_generator.num_classes
    input_shape = train_generator.image_shape
    class_names = [k for k, v in train_generator.class_indices.items()]
    img_h = input_shape[0]
    img_w = input_shape[1]
    print("Image input %s" % str(input_shape))
    print("Classes: %r" % class_names)
    print('Loaded %d training samples from  %d classes.' % (n_samples, n_classes))
    print('Loaded %d test samples from %d classes.' % (validation_generator.n, validation_generator.num_classes))

if __name__ == "__main__":
    print("Tensorflow version %s" %tf.__version__)
    loadData()