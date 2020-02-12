from sklearn.metrics import *
import string
import re
import math
import numpy as np
import sys
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# to surpress future warnings
warnings.simplefilter(action='ignore', category=UserWarning)
# import Use_NN as nn


# Pandas Method to read our CSV to make it easier
def read_csv(filepath):
	# parseDate = ['review_date']
	# dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
	# colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
	# df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
	df_chunk = pd.read_csv(filepath, sep=',', header=0, encoding="ISO-8859-1")
	# df_chuck = df_chuck.fillna(0)
	return df_chunk


def manuallyCalculate(predicted4, truth1):
	testY = truth1
	y_pred = predicted4
	tn, fp, fn, tp = confusion_matrix(y_true=testY, y_pred=y_pred).ravel()
	print(tp, fp, fn, tn)
	print("accu,", accuracy_score(testY, y_pred))
	print("f1,", f1_score(testY, y_pred))
	print("recall,", recall_score(testY, y_pred))
	print("precision", precision_score(testY, y_pred))
	print("pos", (tp/(tp+fn)))
	print("neg", (tn/(tn+fp)))


def addForErrorAnalysis(dataFrame, predicted4):
	dataFrame['predicted'] = predicted4
	print("OUTPUT ANALYSIS")
	dataFrame.to_csv('output_error_analysis.csv')


# Main Method
if __name__ == '__main__':
	print('Number of arguments:', len(sys.argv), 'arguments.')
	print('Argument List:', str(sys.argv))
	if len(sys.argv[1]) > 1:
		predicted = np.load(sys.argv[1], allow_pickle=True)
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit()  # force exit
	if len(sys.argv[2]) > 1:
		truth = np.load(sys.argv[2], allow_pickle=True)
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit()  # force exit
	if len(sys.argv[3]) > 1:
		df = read_csv(sys.argv[3])
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit()  # force exit
	manuallyCalculate(predicted,truth)
	addForErrorAnalysis(df,predicted)

 
