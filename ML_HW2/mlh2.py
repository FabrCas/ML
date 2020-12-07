import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
                         Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers

datadir = "Dataset"
models_dir = "Models"
results_dir = "Results"

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
        target_size= (118, 224),
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

    #print a random samples
    n_ransamples = 2
    x, y = train_generator.next()

    for i in range(0, n_ransamples):
        image = x[i]
        label = y[i].argmax()  # categorical from one-hot-encoding
        print(class_names[label])
        plt.imshow(image)
        plt.show()

    return  input_shape, n_classes

def saveModel(model, problem, ):
    filename = os.path.join(models_dir, '%s.h5' % problem)
    model.save(filename)
    print("\nModel saved on file %s\n" % filename)

def loadModel()

def createCNN(input_shape, n_classes):
    model = Sequential(name="FabrizioNet")
    # L1 -> Convolutional Layer
    model.add(Conv2D(10, kernel_size=(6, 6), strides=(1, 1), activation='relu', input_shape=input_shape, padding='valid'))
    # L2 -> Pooling (MAX)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # L3 -> Convolutional Layer
    model.add(Conv2D(filters=10, kernel_size=(6, 6), activation="relu"))
    # L4 -> Convolutional Layer
    model.add(Conv2D(filters=30, kernel_size=(3, 3)))
    # L5 -> Convolutional Layer
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    # L6 -> Pooling (MAX)
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    # L7 -> flatter layer
    model.add(Flatten())
    # L8 -> Dropout
    model.add(Dropout(0.2))
    # L9 -> dense layer
    model.add(Dense(80, activation='relu'))
    # L10 -> Dropout
    model.add(Dropout(0.2))
    # L11 -> dense layer
    model.add(Dense(50, activation='relu'))
    # L12 -> output layer
    model.add(Dense(n_classes, activation='softmax'))

    optimizer = "adam"
    model.compile(loss= "categorical_crossentropy", optimizer = optimizer, metrics=['accuracy'])
    model.summary()
    return  model

def createTransferLearningModel(input_shape, n_classes):
    pass #todo



if __name__ == "__main__":
    print("Tensorflow version %s" %tf.__version__)
    i_shape, n_classes = loadData()
    type = "CnnScratch"  #CnnScratch or TransferLearning
    cnn_model = createCNN(i_shape, n_classes)

    saveModel(cnn_model, "chosen one")
