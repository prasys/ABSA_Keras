import os
import nltk
import numpy as np
import pandas as pd
from collections import Counter
import spacy
import sys
from spacy.tokenizer import Tokenizer
import re
import locale
#from pandarallel import pandarallel #make things go fasterrr
from sys import platform
CORES = 4
isFirstTime = True
c = Counter()


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

def spacyTokenizer(text,useNLPObj=False,isFirstTime=False):
    # if isFirstTime and useNLPObj:       
    #     nlp = spacy.load("en_core_web_sm")
    #     print("Load Spacy")
    #     nlp.tokenizer = Tokenizer(nlp.vocab) #lod our customized tokenizer overwritten method
    #     isFirstTime  = False
    text = text.lower()
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct is False:
            if token.orth_ == 've': #special handling case
                tokens.append("'ve")
            elif token.orth_ == "  ":
                tokens.append(" ")
            else:
                tokens.append(token.orth_)
    return tokens

def LemenSpacy(text,useNLPObj=False,isFirstTime=False):
    # if isFirstTime and useNLPObj:       
    #     nlp = spacy.load("en_core_web_sm")
    #     print("Load Spacy")
    #     nlp.tokenizer = Tokenizer(nlp.vocab) #lod our customized tokenizer overwritten method
    #     isFirstTime  = False
    text = text.lower()
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct is False:
            if token.orth_ == 've': #special handling case
                tokens.append("'ve")
            elif token.orth_ == "  ":
                tokens.append(" ")
            else:
                print(token.lemma)

                if token.lemma_ == '-PRON-':
                    tokens.append(token.orth_)
                else:
                    tokens.append(token.lemma_)
    return tokens



def addToTotalCounter(tokenizer):
	c.update(tokenizer)

def doItAll(text):
	tokens = spacyTokenizer(text)
	addToTotalCounter(tokens)


file_path = './raw_data/alta/train_22.csv'
nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = Tokenizer(nlp.vocab)
if ('csv' in file_path):
    print("found CSV")
    df = pd.read_csv(file_path, sep=',', header=0,encoding = "ISO-8859-1") #read the file here
elif ('xlsx' in file_path):
    print("found XLSX")
    df = pd.read_excel(file_path, sheet_name='Sheet1')

#https://medium.com/@awantikdas/a-comprehensive-naive-bayes-tutorial-using-scikit-learn-f6b71ae84431

df['Comment'] = df['Comment'].str.lower() # make it lower
df['Comment'] = df['Comment'].apply(scrub_words) #clean up
df['Comment'].apply(doItAll)
b = pd.DataFrame(list(c.items()))
#b = pd.DataFrame.from_dict(c,orient='index') #counter done
b.to_csv('alta_total.csv',index=False)






