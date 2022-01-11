import matplotlib.pyplot as plt
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import pandas as pd

import config

def predicts(X_test, Y_test):
    # работает Только tf 2.0
    saved_model = tf.keras.models.load_model(config.PATH_MODEL)
    #saved_model = tf.saved_model.load(config.PATH_MODEL)
    #saved_model.summary()
    predictions = saved_model(X_test)
    pred = []
    
    print("predictions.shape", predictions.shape)

    Y_val_categorical = to_categorical(Y_test)
    scores = saved_model.evaluate(X_test, Y_val_categorical, verbose=1, )
    
    print("inf/loss: ", scores[0])
    print("inf/acc:", scores[1])
    for p in predictions:
        pred.append(np.argmax(p))

    print("np.shape(Y_test)", np.shape(Y_test))
    print("np.shape(pred))", np.shape(pred))

    cnf_matrix = confusion_matrix(Y_test, pred, labels=[0,1,2,3,4])
    return cnf_matrix, scores


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.inferno_r):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #plt.savefig(config.PATH_IMAGE_CONFUSION_MATRIX)

    #plt.cla()
    return plt.figure()

def df_confusion_matrix(cm, classes, normalize=True):
    """
    This function df the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')    
    df = pd.DataFrame(data=cm, index=classes, columns=classes)
    return df
