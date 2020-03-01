import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) # to surpress future warnings
import pandas as pd
import sys
import numpy as numpy
import math
import gensim
from pprint import pprint
from string import ascii_lowercase
#import Use_NN as nn
import re
import string
from bert_serving.client import BertClient
#import mxnet as mx



# Pandas Method to read our CSV to make it easier
def read_csv(filepath):
    #parseDate = ['review_date']
    #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    #colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
    #df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
    df_chunk = pd.read_csv(filepath, sep=',', header=0,encoding = "ISO-8859-1")
    #df_chuck = df_chuck.fillna(0)
    return df_chunk






#Main Method
if __name__ == '__main__':
	print ('Number of arguments:', len(sys.argv), 'arguments.')
	print ('Argument List:', str(sys.argv))
	if len(sys.argv[2]) > 1:
		df = read_csv(sys.argv[2])
		CommentList = df['Comment'].tolist() # pick the item/column that we want to do BERT embeddings
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit() #force exit
	CommentList = df['Comment'].tolist() # pick the item/column that we want to do BERT embeddings
	print("Start BERT Client")
	bc = BertClient()
	output = bc.encode(CommentList) # enode the list
	print(output.shape)
	numpy.save('output.npy', output, allow_pickle=True) #save the model so that we can use it later for classification and other tasks



	#print(len(results))
	toNumpy = []

	for result in results:
		array = numpy.asarray(result[1])
		mean_array = numpy.mean(array,axis=0)  #convert them into numpy mean array for it
		toNumpy.append(mean_array) 

	output = numpy.array(toNumpy) # convert it to array
	print(output.shape)
	numpy.save('output_alta.npy', output, allow_pickle=True) #save the model so that we can use it later for classification and other tasks


 
