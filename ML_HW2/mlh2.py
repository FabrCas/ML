import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,\
                         Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix
from keras import applications
from keras import callbacks

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
    n_ransamples = 0
    x, y = train_generator.next()

    for i in range(0, n_ransamples):
        image = x[i]
        label = y[i].argmax()  # categorical from one-hot-encoding
        print(class_names[label])
        plt.imshow(image)
        plt.show()

    return  input_shape, n_classes, class_names, train_generator, validation_generator, train_datagen

def saveModel(model, problem):
    filename = os.path.join(models_dir, '%s.h5' % problem)
    model.save(filename)
    print("\nModel saved on file %s\n" % filename)

def saveHistory(history, problem ):
    filename = os.path.join(results_dir, '%s.hist' % problem)
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
    print("\nHystory saved on file %s\n" % filename)

def loadHistory(problem):
   filename = os.path.join(results_dir, '%s.hist' % problem)
   with open(filename, 'rb') as f:
       try:
           history = pickle.load(f)
           print("the history " + filename+ " has been loaded!")
           print(history)
           plot_history(history, "CNN_hw2")
       except OSError:
           print("the history " + filename + " is not present!")
           history = None
   return history


def loadModel(problem):
    filename = os.path.join(models_dir, '%s.h5' % problem)
    try:
        model = load_model(filename)
        print("the model " + filename+ " has been loaded!")
        model.summary()
    except OSError:
        print("the model " + filename + " is not present!")
        model = None
    return model


def createCNN(input_shape, n_classes):
    model = Sequential(name="FabrizioNet")
    # L1 -> Convolutional Layer
    model.add(Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding='valid'))
    # L2 -> Convolutional Layer
    model.add(Conv2D(filters=30, kernel_size=(5, 5), activation="relu"))
    # L3 -> Pooling (MAX)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # L4 -> Convolutional Layer
    model.add(Conv2D(filters=40, kernel_size=(3, 3)))
    # L5 -> Pooling (MAX)
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    # L6 -> flatter layer
    model.add(Flatten())
    # L7 -> dense layer
    model.add(Dense(100, activation='relu'))
    # L8 -> Dropout
    model.add(Dropout(0.2))
    # L9 -> dense layer
    model.add(Dense(60, activation='relu'))
    # L10 -> output layer
    model.add(Dense(n_classes, activation='softmax'))

    optimizer = "adam"
    model.compile(loss= "categorical_crossentropy", optimizer = optimizer, metrics=['accuracy'])
    model.summary()
    return  model

def createTransferLearningModel(input_shape, n_classes):
    model_tl = applications.ResNet50V2(weights="imagenet", include_top= False, input_shape= input_shape)
    # setting not trainable layers
    not_trainable = len(model_tl.layers())-1
    for layer in model_tl.layers()[:not_trainable]:
        layer.trainable = False


def train(model, train_generator, validation_generator, type):
    steps_per_epoch = train_generator.n // train_generator.batch_size
    val_steps = validation_generator.n // validation_generator.batch_size + 1
    history = None
    if type == "CnnScratch":
        try:
            history = model.fit(train_generator, epochs=100, \
                                          steps_per_epoch=steps_per_epoch, \
                                          validation_data=validation_generator, \
                                          validation_steps=val_steps)
            saveHistory(history, "CNN_hw2")
        except KeyboardInterrupt:
            pass
    else:
        pass
        #todo
        #saveHistory(history, "CNN_hw2_tl")

def evaluation(model, validation_generator, class_names):
    # accuracy/loss
    val_steps = validation_generator.n // validation_generator.batch_size + 1
    loss, acc = model.evaluate(validation_generator, steps=val_steps)
    print('Test loss: %f' % loss)
    print('Test accuracy: %f' % acc)
    # Precision/Recall/f-score
    predictions = model.predict(validation_generator, steps=val_steps)
    y_predicted = np.argmax(predictions, axis=1)
    y_test = validation_generator.classes
    print(classification_report(y_predicted, y_test, labels=None, target_names=class_names, digits=3))
    # confusion matrix
    cm = confusion_matrix(y_test, y_predicted)
    conf = []  # data structure for confusions: list of (i,j,cm[i][j])
    for i in range(0, cm.shape[0]):
        for j in range(0, cm.shape[1]):
            if (i != j and cm[i][j] > 0):
                conf.append([i, j, cm[i][j]])

    col = 2
    conf = np.array(conf)
    conf = conf[np.argsort(-conf[:, col])]  # decreasing order by 3-rd column (i.e., cm[i][j])

    print('%-16s     %-16s  \t%s \t%s ' % ('True', 'Predicted', 'errors', 'err %'))
    print('------------------------------------------------------------------')
    for k in conf:
        print('%-16s ->  %-16s  \t%d \t%.2f %% ' % (
        class_names[k[0]], class_names[k[1]], k[2], k[2] * 100.0 / validation_generator.n))


def plot_history(history,name):
    print("********************** plotting the history *********************************")
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(name + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(name + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def blindtest(model, train_datagen):
    batch_size = 64
    blind_test_generator = train_datagen.flow_from_directory(
        directory=blindtest,  # same directory as training data
        target_size=(118, 224),
        batch_size=batch_size,
        shuffle=False,
        class_mode=None)
    predictions = model.predict_generator(blind_test_generator)

    y_predicted = np.argmax(predictions, axis=1)
    print(y_predicted)


if __name__ == "__main__":
    print("Tensorflow version %s" %tf.__version__)
    i_shape, n_classes, class_names, train_generator, validation_generator, train_datagen = loadData()
    type = "CnnScratch"  # CnnScratch or TransferLearning
    if type == "CnnScratch":
        model = loadModel("CNN_hw2")
        if model == None:
            model = createCNN(i_shape, n_classes)
            train(model, train_generator, validation_generator, type)
            saveModel(model, "CNN_hw2")
    else:
        model = loadModel("CNN_hw2_tl")
        if model == None:
            model = createTransferLearningModel(i_shape,n_classes )
            train(model, train_generator, validation_generator, type)
            saveModel(model, "CNN_hw2_tl")


    #evaluation(model, validation_generator, class_names)
    history = loadHistory("CNN_hw2")
    #blindtest(model, train_datagen)





