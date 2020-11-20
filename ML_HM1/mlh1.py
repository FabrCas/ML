import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.feature_extraction.text import *
from sklearn.utils.multiclass import unique_labels

DATASET_PATH = "noduplicatedataset.json"
BLINDTEST_PATH = "nodupblindtest.json"

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
    class_names = np.array([str(c) for c in y_all])
    class_names = pd.DataFrame(class_names).drop_duplicates()
    class_names = np.squeeze(np.asarray(class_names))
    print(class_names)
    return x_all, y_all, class_names

def load_blindTest():
    data = pd.read_json(BLINDTEST_PATH, lines=True)
    x_all = data.lista_asm
    print("********* loading the blindset ********************")
    print("Number of blind samples: %s" % str(x_all.shape[0]))
    # print a random sample
    id = random.randrange(0, x_all.shape[0])
    print("random sampling: ")
    print("x %d = %r" % (id, x_all[id]))
    return x_all


def pre_process_data(x_all):

   #for i,val in enumerate(x_all):
   #    print("\n\n" + str(val))
   #    if i >= 100:
   #        break

   #from x_all -> X

    new_x = []

    registers = ["rax","eax", "ax", "al",
                "rcx" ,"ecx", "cx", "cl",
                "rdx" ,"edx", "dx", "dl",
                "rbx" ,"ebx", "bx", "bl",
                "rsi" ,"esi", "si", "sil",
                "rdi" ,"edi", "di", "dil",
                "rsp" ,"esp", "sp", "spl",
                "rbp" ,"ebp", "bp", "bpl",
                "r8" ,"r8d", "r8w", "r8b",
                "r9" ,"r9d", "r9w", "r9b",
                "r10" ,"r10d", "r10w" ,"r10b",
                "r11" ,"r11d", "r11w" ,"r11b",
                "r12" ,"r12d", "r12w" ,"r12b",
                "r13" ,"r13d", "r13w" ,"r13b",
                "r14" ,"r14d", "r14w" ,"r14b",
                "r15" ,"r15d", "r15w" ,"r15b"]

#
    mathOp = []
    vecmatOp = []
    for asm_list in x_all:
        dict = {}
        dict["length"] = len(asm_list)
        dict["n_for"] = 0; dict["n_if"] = 0; dict["n_xor"] = 0; dict["n_shift"] = 0; dict["n_bitwise"] = 0
        dict["n_helper"] = 0; dict["n_move"] = 0; dict["n_compare"] = 0; dict["n_floatIstruc"] = 0;
        dict["n_mathOp"] = 0; dict["n_vecmatOp"] = 0; dict["n_xmm"] = 0 ; dict["n_swapMem"] = 0

        new_x.append(dict)
    return new_x

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
    print("shape x train" + str(x_train.shape[1]))
    model.fit(xtrain, ytrain)
    return model


def evaluation(model, x_test, y_test):
    print("********* Evaluation of the classification ********************")
    # plotting the confusion matrix
    plot_confusion_matrix(model, x_test, y_test, normalize= "true")
    plt.show()
    # print the classification report
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    n_solution = 2
    model = None
    x_all, y_all, class_names = load_dataset()
    pre_process_data(x_all)
    if n_solution == 1:     # first solution
        x_all = vectorization(x_all)
        x_train, x_test, y_train, y_test = split_data(x_all,y_all)
        model = create_model(x_train,y_train)
        evaluation(model,x_test,y_test)
    else:  # second solution
        pass

    #evaluation of the blind test
   # x_blind = load_blindTest()
   # x_blind = vectorization(x_blind)
   # print(x_blind.shape[1])
   # y_blindPredict = model.predict(x_blind)
    #for y in y_blindPredict:
    #    print(y)




