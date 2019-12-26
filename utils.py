# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: utils.py

@time: 2019/1/5 10:03

@desc:

"""

import pickle
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))


def pickle_dump(obj, file_path):
    pickle.dump(obj, open(file_path, 'wb'))


def get_score_senti(y_true, y_pred):
    """
    return score for predictions made by sentiment analysis model
    :param y_true: array shaped [batch_size, 3]
    :param y_pred: array shaped [batch_size, 3]
    :return:
    """
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print('acc:', acc)
    print('macro_f1:', f1)
    return acc, f1

def get_confusion_matrix(y_true,y_pred):

    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    tn, fp , fn , tp = confusion_matrix(y_pred, y_true).ravel()
    print('DSC:',calcDiceScore(tp,fp,fn))
    print('tp:',tp)
    print('fp:',fp)
    print('fn:',fn)
    print('tn:',tn)
    return tp,fp,fn,tn


def calcDiceScore(tp,fp,fn):
	"""
	returns DSC Score for the sample. Since it's in ordered status , it would be 
	:param y_true: array shaped [batch_size, 3]
	:param y_pred: array shaped [batch_size, 3]
	:return:
	"""
	return (2*tp/(2*tp+fp+fn))











