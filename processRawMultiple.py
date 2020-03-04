import process_raw as pr
import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.model_selection import train_test_split

def test_train_split(embeddings,seed=42):
	trainX, testX, = train_test_split(embeddings,test_size=0.30,random_state=seed)
	return(trainX,testX)


if __name__ == '__main__':
	filePath = './data/books'
	df_train = pd.read_csv('./raw_data/books/BooksTrain2.csv')
	df_test = pd.read_csv('./raw_data/books/BooksTest.csv')
	frames = [df_train, df_test]
	result = pd.concat(frames)

	file = open('randSeed.pkl', 'rb') # load our random seed generator
	data = pickle.load(file)
	for seed in data:
		trainX,testX = test_train_split(result,seed=seed)
		seedPath = filePath + "/" + str(seed)
		pr.process_pandas2(trainX, is_train_file=True, save_folder=seedPath,isClean=True)
		src = seedPath + "/" + "output.csv"
		dst = seedPath + "/" + "output_train.csv"
		print("Renaming train file")
		os.rename(src, dst)
		pr.process_pandas2(testX, is_train_file=True, save_folder=seedPath,isClean=True,countSentence=True)
		src = seedPath + "/" + "output.csv"
		dst = seedPath + "/" + "output_test.csv"
		print("Renaming test file")
		os.rename(src, dst)
		src = seedPath + "/" + "output_train.csv"
		dst = seedPath + "/" + "train.csv"
		print("Renaming others file")
		os.remove(dst)
		os.rename(src, dst)
		src = seedPath + "/" + "output_test.csv"
		dst = seedPath + "/" + "test.csv"
		print("Renaming others file")
		os.remove(dst)
		os.rename(src, dst)



