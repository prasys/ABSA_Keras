# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#https://www.kaggle.com/tanumoynandy/sarcasm-detection-rnn-lstm <-- HOME WORK
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D , Dropout , Conv1D , MaxPooling1D , Activation , GlobalMaxPooling1D , Input, Lambda, Dense
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model
import keras.backend as K
from collections import Counter
import tensorflow_hub as hub
import tensorflow as tf
import sys
import re
#url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

# ClassifyIt = True

max_features = 240 
def read_csv(filepath):
    df_chunk = pd.read_csv(filepath)
    #df_chuck = df_chuck.fillna(0)
    return df_chunk


def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)))


def processEncoder(inputSequence,URL,isTrainable,nameThingy):
    url = URL
    embed = hub.Module(url,trainable=isTrainable)
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())  
        session.run(tf.tables_initializer())
        tweet_embeddings = embed(inputSequence.tolist())
        tweet_embeddings = tweet_embeddings.eval()
        #session.run(tweet_embeddings)
        #tweet_embeddings = np.array(tweet_embeddings)
        embeeddingName = nameThingy + ".npy"
        np.save(embeeddingName,tweet_embeddings)


def preProcessStuff(inputFile,ColumnName):
    df = read_csv(inputFile)
    df[ColumnName] = df[ColumnName].str.lower() # make them lower
    totalNum = df[ColumnName].str.len() # length of it
    avg = np.mean(totalNum)
    max_len = int(avg)
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df[ColumnName].values)
    X = tokenizer.texts_to_sequences(df[ColumnName].values)
    X = pad_sequences(X,maxlen=max_len)
    x = df[ColumnName]
    return x


max_features = 240 #length of the maximum ones
if len(sys.argv[1]) > 1:
    df = read_csv(sys.argv[1])
else:
    print("Error - NO INPUT FILE GIVEN")
    exit()
if len(sys.argv[2]) > 1:
    nameThingy = sys.argv[2]
else:
    print("Error - NO OUTPUT FILE GIVEN")
    exit()



if __name__ == '__main__':
    if len(sys.argv[1]) > 1: # File Name 
        CSVFileName = sys.argv[1]
    else:
        print("Error - NO INPUT FILE GIVEN")
        exit()

    if len(sys.argv[2]) > 1: # URL to download sentence Encoder 
        URLString = sys.argv[2]
    else:
        print("Error - NO INPUT FILE GIVEN")
        exit()

    if len(sys.argv[3]) > 1: #OUTPUT FILE NAME
        outputFileName = sys.argv[3]
    else:
        print("NO OUTPUT FILE GIVEN")
        exit()

    if len(sys.argv[4]) > 1: #OUTPUT FILE NAME
        columnName = sys.argv[4]
    else:
        print("NO OUTPUT FILE GIVEN")
        exit()

    if len(sys.argv[5]) > 1: # TO DECIDE TO TRAIN or NOT , NEED TO ADD A CATCH CLAUSE TO ALLOW IT TO CATCH EXCEPTIONS FOR IT
        binaryDecider = sys.argv[5]
    else:
        binaryDecider = True


    processedStuff = preProcessStuff(CSVFileName,columnName)
    processEncoder(processedStuff,URLString,binaryDecider,outputFileName)




