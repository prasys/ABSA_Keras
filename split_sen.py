import os
import nltk
import numpy as np
import pandas as pd
from collections import Counter
import spacy
import sys
from spacy.tokenizer import Tokenizer
import locale
#from pandarallel import pandarallel #make things go fasterrr
from sys import platform
CORES = 4
isFirstTime = True
totalstuff[]




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


def addToTotalCounter(tokenizer):
	if isFirstTime:
		c = Counter(tokenizer)
	else:
		c.update(tokenizer)

	return c



