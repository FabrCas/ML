import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import *
from sklearn.utils.multiclass import unique_labels

DATASET_PATH = "noduplicatedataset.json"
BLINDTEST_PATH = "nodupblindtest.json"

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def load_dataset():
    data = pd.read_json(DATASET_PATH, lines=True)
    x_all = data.lista_asm
    y_all = data.semantic
    print("********* loading the dataset ********************")
    print("Number of input samples X: %s" % str(x_all.shape[0]))
    print("Number of output classifications Y: : %s" % str(y_all.shape[0]))
    # print a random sample
    id = random.randrange(0, x_all.shape[0])
    print("random sampling: ")
    print("x %d = %r" % (id, x_all[id]))
    print("y %d = %r" % (id, y_all[id]))
    pre_process_data(x_all)
    return x_all, y_all

def load_blindTest():
    pass

def pre_process_data(x_all):
    # from x_all -> X
    # X can be the number of
    pass

def vectorization(x):
    vectorizer = CountVectorizer(stop_words='english')
    x_all = vectorizer.fit_transform(x)
    return x_all

def split_data(x,y):
    print("********* Splitting the dataset ********************")
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size= 0.2, random_state= 117)
    print("Size of training set: %d" % xtrain.shape[0])
    print("Size of test set: %d" % xtest.shape[0])
    return xtrain, xtest, ytrain, ytest


def create_model(xtrain, ytrain):
    print("********* creating the SVM model (linear kernel) ********************")
    model = svm.SVC(kernel='linear', gamma='scale')
    model.fit(xtrain, ytrain)


def evaluation():
    pass

if __name__ == '__main__':
    x_all, y_all = load_dataset()
    x_all = vectorization(x_all)
    x_train, x_test, y_train, y_test = split_data(x_all,y_all)


