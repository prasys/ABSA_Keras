import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) # to surpress future warnings
import pandas as pd
import sys
import numpy as numpy
#import math
#import gensim
#from pprint import pprint
#from string import ascii_lowercase
#import Use_NN as nn
#import re
#import string
#from bert_serving.client import BertClient
#import mxnet as mx
from embedding_as_service.text.encode import Encoder  



# Pandas Method to read our CSV to make it easier
def read_csv(filepath):
	#parseDate = ['review_date']
	#dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
	#colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
	#df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
	df_chunk = pd.read_csv(filepath, sep=',', header=0,encoding = "ISO-8859-1")
	#df_chuck = df_chuck.fillna(0)
	return df_chunk

def createEmbedding(text,embedding,model,outPutFileName):
	en = Encoder(embedding=embedding, model=model, max_seq_length=31)
	output = en.encode(text,pooling='reduce_mean')
	print(output.shape)
	outputFileName = outPutFileName
	numpy.save(outputFileName, output, allow_pickle=True) #save the model so that we can use it later for classification and other tasks




#Main Method
if __name__ == '__main__':
	print ('Number of arguments:', len(sys.argv), 'arguments.')
	print ('Argument List:', str(sys.argv))
	if len(sys.argv[1]) > 1:
		df = read_csv(sys.argv[1])
		CommentList = df['Comment'].tolist() # pick the item/column that we want to do BERT embeddings
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit() #force exit
	if len(sys.argv[2]) > 1:
		embedding = sys.argv[2]
		
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit() #force exit
	if len(sys.argv[3]) > 1:
		model = sys.argv[3]
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit() #force exit

	if len(sys.argv[4]) > 1:
		outPutFileName = sys.argv[4]
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit() #force exit		
	
	CommentList = df['Comment'].tolist() # pick the item/column that we want to do BERT embeddings
	print("Start Embedding Service Client")
	createEmbedding(CommentList,embedding,model,outPutFileName)





 
