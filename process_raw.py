# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: process_raw.py

@time: 2019/1/5 17:05

@desc: process raw data

"""

import os
import codecs
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import random
import re
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import numpy as np
from sklearn.utils import resample # to handle resampling technique to resample the minority class to see if it works
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

RANDOMSTATE = 21

def initNLP():
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = Tokenizer(nlp.vocab)
    return nlp


def handle_imbalance(df,label):
    # Group the names by label , and check which one is excess , remove the extras and untill we get it - and then return the balanced dataset to test it up
    g = df.groupby(label)
    k = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    return k

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


def process_xml(file_path, is_train_file, save_folder):
    content_term, aspect_term, sentiment_term, start, end = list(), list(), list(), list(), list() # data for aspect term
    content_cate, aspect_cate, sentiment_cate = list(), list(), list()      # data for aspect category
    polarity = {'negative': 0, 'neutral': 1, 'positive': 2}

    tree = ET.parse(file_path)
    root = tree.getroot()
    for sentence in root:
        text = sentence.find('text').text.lower()
        for asp_terms in sentence.iter('aspectTerms'):
            for asp_term in asp_terms.iter('aspectTerm'):
                if asp_term.get('polarity') in polarity:
                    _text = text
                    _start = int(asp_term.get('from'))
                    _end = int(asp_term.get('to'))
                    _aspect = asp_term.get('term').lower()
                    _sentiment = polarity[asp_term.get('polarity')]
                    if _start > 0 and text[_start - 1] != ' ':
                        _text = text[:_start] + ' ' + text[_start:]
                        _start += 1
                        _end += 1
                    if _end < len(_text) and _text[_end] != ' ':
                        _text = _text[:_end] + ' ' + _text[_end:]
                    if _text[_start:_end] != _aspect:
                        raise Exception('{}=={}=={}'.format(_text, _text[_start:_end], _aspect))
                    content_term.append(_text)
                    aspect_term.append(_aspect)
                    sentiment_term.append(_sentiment)
                    start.append(_start)
                    end.append(_end)
        for asp_cates in sentence.iter('aspectCategories'):
            for asp_cate in asp_cates.iter('aspectCategory'):
                if asp_cate.get('polarity') in polarity:
                    content_cate.append(text)
                    aspect_cate.append(asp_cate.get('category'))
                    sentiment_cate.append(polarity[asp_cate.get('polarity')])

    if not os.path.exists(os.path.join(save_folder, 'term')):
        os.makedirs(os.path.join(save_folder, 'term'))

    if not is_train_file:
        test_data = {'content': content_term, 'aspect': aspect_term, 'sentiment': sentiment_term,
                     'from': start, 'to': end}
        test_data = pd.DataFrame(test_data, columns=test_data.keys())
        test_data.to_csv(os.path.join(save_folder, 'term/test.csv'), index=None)
    else:
        train_content, valid_content, train_aspect, valid_aspect, train_senti, valid_senti, train_start, valid_start, \
            train_end, valid_end = train_test_split(content_term, aspect_term, sentiment_term, start, end, test_size=0.1)
        train_data = {'content': train_content, 'aspect': train_aspect, 'sentiment': train_senti,
                      'from': train_start, 'to': train_end}
        train_data = pd.DataFrame(train_data, columns=train_data.keys())
        train_data.to_csv(os.path.join(save_folder, 'term/train.csv'), index=None)
        valid_data = {'content': valid_content, 'aspect': valid_aspect, 'sentiment': valid_senti,
                      'from': valid_start, 'to': valid_end}
        valid_data = pd.DataFrame(valid_data, columns=valid_data.keys())
        valid_data.to_csv(os.path.join(save_folder, 'term/valid.csv'), index=None)

    if len(content_cate) > 0:
        if not os.path.exists(os.path.join(save_folder, 'category')):
            os.makedirs(os.path.join(save_folder, 'category'))

        if not is_train_file:
            test_data = {'content': content_cate, 'aspect': aspect_cate, 'sentiment': sentiment_cate}
            test_data = pd.DataFrame(test_data, columns=test_data.keys())
            test_data.to_csv(os.path.join(save_folder, 'category/test.csv'), index=None)
        else:
            train_content, valid_content, train_aspect, valid_aspect, \
                train_senti, valid_senti = train_test_split(content_cate, aspect_cate, sentiment_cate, test_size=0.1)
            train_data = {'content': train_content, 'aspect': train_aspect, 'sentiment': train_senti}
            train_data = pd.DataFrame(train_data, columns=train_data.keys())
            train_data.to_csv(os.path.join(save_folder, 'category/train.csv'), index=None)
            valid_data = {'content': valid_content, 'aspect': valid_aspect, 'sentiment': valid_senti}
            valid_data = pd.DataFrame(valid_data, columns=valid_data.keys())
            valid_data.to_csv(os.path.join(save_folder, 'category/valid.csv'), index=None)


def process_twitter(file_path, is_train_file, save_folder):
    polarity = {'-1': 0, '0': 1, '1': 2}
    content, aspect, sentiment, start, end = list(), list(), list(), list(), list()
    with codecs.open(file_path, 'r', encoding='utf8')as reader:
        lines = reader.readlines()
        for i in range(0, len(lines), 3):
            _content = lines[i].strip().lower()
            _aspect = lines[i+1].strip().lower()
            _sentiment = lines[i+2].strip().lower()
            _start = _content.find('$t$')
            _end = _start + len(_aspect)
            content.append(_content.replace('$t$', _aspect))
            aspect.append(_aspect)
            sentiment.append(polarity[_sentiment])
            start.append(_start)
            end.append(_end)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not is_train_file:
        test_data = {'content': content, 'aspect': aspect, 'sentiment': sentiment, 'from': start, 'to': end}
        test_data = pd.DataFrame(test_data, columns=test_data.keys())
        test_data.to_csv(os.path.join(save_folder, 'test.csv'), index=None)
    else:
        train_content, valid_content, train_aspect, valid_aspect, train_senti, valid_senti, train_start, valid_start, \
            train_end, valid_end = train_test_split(content, aspect, sentiment, start, end, test_size=0.1)
        train_data = {'content': train_content, 'aspect': train_aspect, 'sentiment': train_senti,
                      'from': train_start, 'to': train_end}
        train_data = pd.DataFrame(train_data, columns=train_data.keys())
        train_data.to_csv(os.path.join(save_folder, 'train.csv'), index=None)
        valid_data = {'content': valid_content, 'aspect': valid_aspect, 'sentiment': valid_senti,
                      'from': valid_start, 'to': valid_end}
        valid_data = pd.DataFrame(valid_data, columns=valid_data.keys())
        valid_data.to_csv(os.path.join(save_folder, 'valid.csv'), index=None)

## TO-DO IMPLEMENT THE PROCESS TO MAKE RAW TO UNDERSTAND IT FURTHER
def process_pandas(file_path, is_train_file, save_folder):
    df = pd.read_csv(file_path, sep=',', header=0,encoding = "ISO-8859-1") #read the file here
    df['Comment'] = df['Comment'].str.lower()
    df['Comment'] = df['Comment'].apply(cleanSpecialCharacters)
    df['Prediction'] = df['Prediction'].astype(str) #make them as str
    df['Prediction'] = df['Prediction'].str.lower()
    df['Prediction'] = df['Prediction'].apply(cleanSpecialCharacters)
    text = []
    target = []
    sentScore = 1
    leftIndex = []
    rightIndex = []
    sentiment = []
    dfObj = pd.DataFrame(columns=['content', 'aspect', 'sentiment','from','to'])

    for index, row in df.iterrows():
        if row['Label'] is 0:
            tokens = row['Prediction'].split(' ')
            if len(tokens) > 1:
                sentScore = 2 # the target has multiple
            else:
                sentScore = 1 # there is only 1 target 
            for token in tokens:
                m = re.search(r'\b'+ re.escape(token) +r'\b', row['Comment'], re.IGNORECASE)

                if m:
                    start_index = m.start()
                else:
                    start_index = row['Comment'].find(token)
                end_index = start_index + len(token)
                text.append(row['Comment'])
                target.append(token)
                leftIndex.append(start_index)
                rightIndex.append(end_index)
                sentiment.append(sentScore)
        else:
            word = random.choice(row['Comment'].split())
            sentScore = 0 # we will set 0 to be very weak
            m = re.search(r'\b'+ re.escape(word) +r'\b', row['Comment'], re.IGNORECASE)
            if m:
                start_index = m.start()
            else:
                start_index = row['Comment'].find(word)
            end_index = start_index + len(word)
            text.append(row['Comment'])
            target.append(word)
            leftIndex.append(start_index)
            rightIndex.append(end_index)
            sentiment.append(sentScore)

    
    dfObj['content'] =text
    dfObj['aspect'] =target
    dfObj['aspect'] = df['aspect'].astype(str)
    dfObj['aspect'] = dfObj['aspect'].str.lower() #make everything lower case 
    dfObj['sentiment'] = sentiment
    dfObj['from'] =leftIndex
    dfObj['to'] = rightIndex
    X_train, X_test = train_test_split(dfObj,test_size=0.33, random_state=42)
    xTest,xValidate = train_test_split(X_test,test_size=0.10, random_state=42)

    X_train.to_csv(os.path.join(save_folder, 'train.csv'), index=None)
    xTest.to_csv(os.path.join(save_folder, 'test.csv'), index=None)
    xValidate.to_csv(os.path.join(save_folder, 'valid.csv'), index=None)


def process_pandas2(file_path, is_train_file, save_folder,isClean=False,countSentence=False):
    if countSentence is True:
        instanceCounter = []
    nlp = initNLP() # start our NLP detection for this.
    if ('csv' in file_path):
        print("found CSV")
        df = pd.read_csv(file_path, sep=',', header=0,encoding = "ISO-8859-1") #read the file here
    elif ('xlsx' in file_path):
        print("found XLSX")
        df = pd.read_excel(file_path, sheet_name='Sheet1')
    else:
        print("Probably a dataframe")
        df = file_path

    if isClean is True:
        print("Applying scrub cleaner")
        df['Comment'] = df['Comment'].apply(scrub_words)
        df['Prediction'] = df['Prediction'].apply(scrub_words)

    df['Comment'] = df['Comment'].str.lower()
   # df['Comment'] = df['Comment'].apply(cleanSpecialCharacters)
    df['Prediction'] = df['Prediction'].astype(str) #make them as str
    df['Prediction'] = df['Prediction'].str.lower()

    #df['Prediction'] = df['Prediction'].apply(cleanSpecialCharacters)
    text = []
    target = []
    sentScore = 0
    wordCount = 0
    leftIndex = []
    rightIndex = []
    sentiment = []
    dfObj = pd.DataFrame(columns=['content', 'aspect', 'sentiment','from','to'])

    for index, row in df.iterrows():
        if (row['Label'] != 3): # if it's all - stupid check / need to refactor this          

            tokens = nlp(row['Comment']) # get the stuff
            truths =  nlp(row['Prediction'])
            isFound = False

            for token in tokens: #tokenize
                if token.is_punct is False: #just to make sure we don't get the dots and other junk
                    if countSentence is True: # this is to determine the no of sentences found inside , it is required later on 
                        wordCount = wordCount+1
                    # print("Current tOKEN",token.orth_)
                    for truth in truths:
                        isFound = False
                        if(truth.orth_ == token.orth_): #if we have found a match with the prediction , we add them in 
                            m = re.finditer(r'\b'+ re.escape(truth.orth_) +r'\b', row['Comment'], re.IGNORECASE) # find all instances
                            mlength = re.findall(r'\b'+ re.escape(truth.orth_) +r'\b', row['Comment'], re.IGNORECASE) # find all instances
                            for indeks in m:
                                if len(mlength) == 1:
                                    start_index = indeks.start()
                                    end_index = start_index + len(truth.orth_)
                                    sentScore = 1
                                    text.append(row['Comment'])
                                    target.append(truth.orth_)
                                    leftIndex.append(start_index)
                                    rightIndex.append(end_index)
                                    sentiment.append(sentScore)
                                    isFound = True
                                    break
                                else:
                                    peekValue = indeks.start() + len(truth.orth_)
                                    endValue = peekValue + 1
                                    if  " " in row['Comment'][peekValue:endValue]:
                                        start_index = indeks.start()
                                        end_index = start_index + len(truth.orth_)
                                        sentScore = 1
                                        text.append(row['Comment'])
                                        target.append(truth.orth_)
                                        leftIndex.append(start_index)
                                        rightIndex.append(end_index)
                                        sentiment.append(sentScore)
                                        isFound = True
                                        break
                                break

                    if(isFound == False):
                        # print("CURRENT Token for FALSE",token.orth_)
                        sentScore = 0
                        m = re.finditer(r'\b'+ re.escape(token.orth_) +r'\b', row['Comment'], re.IGNORECASE)
                        mlength = re.findall(r'\b'+ re.escape(token.orth_) +r'\b', row['Comment'], re.IGNORECASE) # find all instances
                        # print("all inside length is",mlength)
                        # print("length is",len(mlength))
                        if(len(mlength)==0):
                            start_index = row['Comment'].find(token.orth_)
                            end_index = start_index + len(token.orth_)
                            target.append(token.orth_)
                            text.append(row['Comment'])
                            leftIndex.append(start_index)
                            rightIndex.append(end_index)
                            sentiment.append(sentScore)
                        for indeks in m:
                            if len(mlength) == 1:
                                start_index = indeks.start()
                                end_index = start_index + len(token.orth_)
                                target.append(token.orth_)
                                text.append(row['Comment'])
                                leftIndex.append(start_index)
                                rightIndex.append(end_index)
                                sentiment.append(sentScore)
                            else:
                                peekValue = indeks.start() + len(token.orth_)
                                endValue = peekValue + 1
                                if  " " in row['Comment'][peekValue:endValue] and " " in row['Comment'][indeks.start()-1:indeks.start()]:
                                    start_index = indeks.start()
                                    end_index = start_index + len(token.orth_)
                                    target.append(token.orth_)
                                    text.append(row['Comment'])
                                    leftIndex.append(start_index)
                                    rightIndex.append(end_index)
                                    sentiment.append(sentScore)
            if countSentence is True:
                instanceCounter.append(wordCount)
                wordCount = 0

    dfObj['content'] =text
    dfObj['aspect'] =target
    dfObj['sentiment'] = sentiment
    dfObj['from'] =leftIndex
    dfObj['to'] = rightIndex
    dfObj = dfObj.drop_duplicates(subset=['content','from','to'])
    dfObj.to_csv(os.path.join(save_folder, 'output.csv'), index=None) # This is needed to be read by the preprocess.py
    if countSentence is True:
        outputarray = np.asarray(instanceCounter)
        np.save(os.path.join(save_folder, 'totalsentence.npy'), outputarray)
        print("Saved Total Sentence to File") 
    if is_train_file is True:
        X_train, X_test = train_test_split(dfObj,test_size=0.30, shuffle=False,random_state=10000) # change the size
        xTest,xValidate = train_test_split(X_test,test_size=0.50, shuffle=False,random_state=10000) # change the size
        X_train.to_csv(os.path.join(save_folder, 'train.csv'), index=None)
        xTest.to_csv(os.path.join(save_folder, 'test.csv'), index=None)
        xValidate.to_csv(os.path.join(save_folder, 'valid.csv'), index=None)

def process_fsauor(file_path, save_path):
    folder = os.path.dirname(save_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    aspect_names = ['location_traffic_convenience', 'location_distance_from_business_district',
                    'location_easy_to_find', 'service_wait_time', 'service_waiters_attitude',
                    'service_parking_convenience', 'service_serving_speed', 'price_level',
                    'price_cost_effective', 'price_discount', 'environment_decoration', 'environment_noise',
                    'environment_space', 'environment_cleaness', 'dish_portion', 'dish_taste', 'dish_look',
                    'dish_recommendation', 'others_overall_experience', 'others_willing_to_consume_again']
    polarity = {-1: 0, 0: 1, 1: 2}
    content, aspect, sentiment = list(), list(), list()
    raw_data = pd.read_csv(file_path, header=0, index_col=0)

    for _, row in raw_data.iterrows():
        text = row['content']
        for aspect_name in aspect_names:
            if row[aspect_name] != -2:
                content.append(text)
                aspect.append(aspect_names)
                sentiment.append(polarity[row[aspect_name]])

    csv_data = {'content': content, 'aspect': aspect, 'sentiment': sentiment}
    csv_data = pd.DataFrame(csv_data, columns=csv_data.keys())
    csv_data.to_csv(save_path, index=0)


def process_bdci(file_path, is_train_file, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    raw_data = pd.read_csv(file_path, header=0, index_col=0)
    if is_train_file:
        polarity = {-1: 0, 0: 1, 1: 2}
        raw_data = raw_data[['content', 'subject', 'sentiment_value']]
        raw_data['sentiment_value'].replace(polarity, inplace=True)
        raw_data.rename(columns={'subject': 'aspect', 'sentiment_value': 'sentiment'})
        train_data, valid_data = train_test_split(raw_data, test_size=0.1)
        train_data.to_csv(os.path.join(save_folder, 'train.csv'), index=None)
        valid_data.to_csv(os.path.join(save_folder, 'valid.csv'), index=None)
    else:
        raw_data[['content']].to_csv(os.path.join(save_folder, 'test.csv'), index=None)


if __name__ == '__main__':
   # process_xml('./raw_data/semeval14_laptop/Laptop_Train_v2.xml', is_train_file=True, save_folder='./data/laptop')
   # process_xml('./raw_data/semeval14_laptop/Laptops_Test_Gold.xml', is_train_file=False, save_folder='./data/laptop')

   # process_xml('./raw_data/semeval14_restaurant/Restaurants_Train_v2.xml', is_train_file=True,
   #             save_folder='./data/restaurant')
   # process_xml('./raw_data/semeval14_restaurant/Restaurants_Test_Gold.xml', is_train_file=False,
   #             save_folder='./data/restaurant')

   # process_twitter('./raw_data/twitter/train.txt', is_train_file=True, save_folder='./data/twitter')
    process_pandas2('./raw_data/alta/train_alta_dataset.csv', is_train_file=True, save_folder='./data/alta2',isClean=True)
 #   process_pandas2('./raw_data/books/book_snippet.xlsx', is_train_file=True, save_folder='./data/books' , isClean=True)
   # process_twitter('./raw_data/twitter/test.txt', is_train_file=False, save_folder='./data/twitter')
    # process_fsauor('./raw_data/fsauor2018/train.csv', save_path='./data/fsauor/train.csv')
    # process_fsauor('./raw_data/fsauor2018/valid.csv', save_path='./data/fsauor/valid.csv')
    # process_fsauor('./raw_data/fsauor2018/test.csv', save_path='./data/fsauor/test.csv')
    #
    # process_bdci('./raw_data/bdci18_car/train.csv', is_train_file=True, save_folder='./data/bdci')
    # process_bdci('./raw_data/bdci18_car/test.csv', is_train_file=True, save_folder='./data/bdci')

