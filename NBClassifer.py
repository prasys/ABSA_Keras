import process_raw as pr
import pandas as pd
import numpy as np
import pickle
import sys
import os
import re
from spacy.tokenizer import Tokenizer
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
from sklearn.naive_bayes import MultinomialNB


def splitThem(embeddings,seed=42):
	text = embeddings['Cleaned']
	truth = embeddings['Label']
	trainX, testX = train_test_split(text,test_size=0.30,random_state=seed)
	trainY, testY = train_test_split(truth,test_size=0.30,random_state=seed)
	return(trainX,testX,trainY,testY)

def cleanSpecialCharacters(text):
    return (re.sub( '[^a-z0-9\']', ' ', text))

def scrub_words(text):
    """Basic cleaning of texts."""
    """Taken from https://github.com/kavgan/nlp-in-practice/blob/master/text-pre-processing/Text%20Preprocessing%20Examples.ipynb """
    
    # remove html markup
    text=re.sub("(<.*?>)","",text)
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    # remove the extra spaces that we have so that it is easier for our split :) Taken from https://stackoverflow.com/questions/2077897/substitute-multiple-whitespace-with-single-whitespace-in-python
    text=re.sub(' +', ' ', text).strip()
    return text

def matrixScoreCalculations(predictedX,actualX,fileName,seed=42):
	with open('out.txt', 'a+') as f:
		tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=actualX, y_pred=predictedX).ravel()
		print(fileName,file=f)
		print(tp,fp,fn,tn,file=f)
		print("Seed Value is",seed,file=f)
		print("Accuracy score", sklearn.metrics.balanced_accuracy_score(predictedX, actualX),file=f)
		print("F1 score", sklearn.metrics.f1_score(predictedX, actualX),file=f)
		# print("Recall score", sklearn.metrics.recall_score(predictedX, actualX),file=f)
		# print("Percision score", sklearn.metrics.precision_score(predictedX, actualX),file=f)
		# pos = (tp/(tp+fn))
		# neg = (tn/(tn+fp))
		# balAcc = (pos + neg)/2
		# print("Balanced Accuracy score", sklearn.metrics.accuracy_score(predictedX, actualX),file=f)
		# print("Positive Score",(tp/(tp+fn)),file=f)
		# print("Negative Score",(tn/(tn+fp)),file=f)
		print("END",file=f)


def LemenSpacy(text,useNLPObj=False,isFirstTime=False):
    text = text.lower()
    doc = nlp(text)
    tokens = []
    k = ' ff'
    for token in doc:
        if token.is_punct is False:
            if token.orth_ == 've': #special handling case
                tokens.append("'ve")
            elif token.orth_ == "  ":
                tokens.append(" ")
            else:
                if token.lemma_ == '-PRON-':
                    tokens.append(token.orth_)
                    k.join(str(token.orth_))
                else:
                    tokens.append(token.lemma_)
                    k.join(str(token.lemma_))
    return (' '.join(tokens)) 
    #return tokens


def performNBClassification(trainX,testX,trainY,testY,seed=42):
	cv = CountVectorizer(binary=False,ngram_range=(1,3)) 
	x_Train = cv.fit_transform(trainX)
	x_test = cv.transform(testX)
	mnb = MultinomialNB()
	mnb.fit(x_Train,trainY)
	mnb_prediction = mnb.predict(x_test)
	with open('out_nb.txt', 'a+') as f:
		tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=testY, y_pred=mnb_prediction).ravel()
		print("Seed Value is",seed,file=f)
		print("Accuracy score", sklearn.metrics.balanced_accuracy_score(mnb_prediction, testY),file=f)
		print("F1 score", sklearn.metrics.f1_score(mnb_prediction, actualX),file=f)


if __name__ == '__main__':
	filePath = './data/alta'
	nlp = spacy.load("en_core_web_sm")
	nlp.tokenizer = Tokenizer(nlp.vocab)
	df_train = pd.read_csv('./raw_data/alta/train_alta_dataset.csv')
	df_test = pd.read_csv('./raw_data/alta/test_alta_dataset.csv')
	frames = [df_train, df_test]
	result = pd.concat(frames)
	result['Comment'] = result['Comment'].str.lower() # make it lower
	result['Comment'] = result['Comment'].apply(scrub_words) #clean up
	result['Cleaned'] = result['Comment'].apply(LemenSpacy) 

	file = open('randSeed.pkl', 'rb') # load our random seed generator
	data = pickle.load(file)
	for val in data:
		trainX,testX,trainY,testY = splitThem(result,val)
		performNBClassification(trainX,testX,trainY,testY,val)
		

