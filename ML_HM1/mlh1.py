import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

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
    x_all = pre_process_data(x_all)
    return x_all, y_all, class_names

def load_blindTest():
    data = pd.read_json(BLINDTEST_PATH, lines=True)
    x_all = data.lista_asm
    id_all =data.id
    print("********* loading the blindset ********************")
    print("Number of blind samples: %s" % str(x_all.shape[0]))
    # print a random sample
    id = random.randrange(0, x_all.shape[0])
    print("random sampling: ")
    print("x %d = %r" % (id, x_all[id]))
    x_all = pre_process_data(x_all)
    return x_all, id_all


def pre_process_data(x_all):
    print("***************** Pre-processing ********************")
    print(type(x_all))
    registers = [
        "rax","eax", "ax", "al",
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
    mathOp = ["inc", "dec", "neg", "leaq", "add", "sub", "imul", "imulq",
              "mulq", "idivq", "divq", "idivl", "divl", "cltd"]
    bitwiseOp = ["not", "or", "and", "xor"]
    shifOp = ["sal", "shl", "sar", "shr"]
    dataMovementOp = ["mov", "push", "pop", "cwtl", "cltq", "cqto"]
    compTestOp = ["cmp", "test","cmpq"]
    externalCall =["call", "ret"]
    forSt = ["loop", "jle", "ja", "jb", "jz", "jmp", "je", "jne", "jnz","js","jns","jg", "jnle", "jge", "jnl", "jl",
             "jnge", "jle", "jng", "ja", "jnbe", "jae", "jnb", "jnae", "jbe", "jna"]
    floatingPoint = ["xmm", "movs", "cvtss2sd", "cvtsi2s", "cvtts", "adds", "subs",
                     "muls", "divs", "maxs", "mins","sqrts", "ucomis"]
    checks= [registers,mathOp,bitwiseOp,shifOp,dataMovementOp,compTestOp,forSt,floatingPoint,externalCall]
    occ_register = 0
    occ_call = 0

    for i,asm_list in enumerate(x_all):
        n_instruction = len(asm_list.split("', '"))
        new_x = str("instruction_n_"+ str(n_instruction) + " ")
        for checkgroup in checks:
            for j, word in enumerate(checkgroup):
                occ = asm_list.count(word)
                if j == 0:
                    occ_register += 1
                if word == "call":
                    occ_call += 1
                if occ >= 0:
                    new_x = new_x + word + str(occ) + " "

        new_x = new_x + "register_uses " + str(occ_register)
        new_x = new_x + "call_uses " + str(occ_call)
        x_all[i] = new_x

    return x_all

def vectorization1(x):
    print("********* Vectorization of data ********************")
    vectorizer = CountVectorizer()
    #vectorizer = HashingVectorizer()

    # Learn the vocabulary dictionary and return document-term matrix.
    # with shape attribute retrieve dimensions
    x_all = vectorizer.fit_transform(x)
    return x_all, vectorizer

def vectorization2(x):
    print("********* Vectorization of data ********************")
    vectorizer = HashingVectorizer(ngram_range=(1,3))

    # Learn the vocabulary dictionary and return document-term matrix.
    # with shape attribute retrieve dimensions
    x_all = vectorizer.fit_transform(x)
    return x_all, vectorizer

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
    return model

def create_model2(xtrain, ytrain):
    print("********* creating the Bernoulli model ********************")
    model = BernoulliNB()
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

def predictionBlindData(model, vectorizer):
    print("********* Prediction for the blind ********************")
    x_blind , ids_blind = load_blindTest()
    x_blind = vectorizer.transform(x_blind)
    y_blindPredict = model.predict(x_blind)

    print("*******************end******************************")
    #for y in y_blindPredict:
    #    print(y)

if __name__ == '__main__':
    n_solution = 3 # change for creating different results
    model = None

    x_all, y_all, class_names = load_dataset()
    if n_solution == 1:                    # first solution
        print("************************** 1° solution *********************************************")
        x_all, vectorizer = vectorization1(x_all)
        x_train, x_test, y_train, y_test = split_data(x_all,y_all)
        model = create_model(x_train,y_train)
        evaluation(model,x_test,y_test)

        #predictionBlindData(model, vectorizer)

    elif(n_solution==2):                # second solution
        print("************************** 2° solution *********************************************")
        x_all, vectorizer = vectorization2(x_all)
        x_train, x_test, y_train, y_test = split_data(x_all,y_all)
        model = create_model(x_train,y_train)
        evaluation(model,x_test,y_test)
    elif (n_solution == 3):             #third solution
        print("************************** 3° solution *********************************************")
        x_all, vectorizer = vectorization1(x_all)
        x_train, x_test, y_train, y_test = split_data(x_all, y_all)
        model = create_model2(x_train, y_train)
        evaluation(model, x_test, y_test)








