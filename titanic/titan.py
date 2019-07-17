#import data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import category_encoders as ce
#import random as rnd
#sns.set()

def run_ml(features):
	#create dataframes from files
	train_df = pd.read_csv('./train.csv')[features]
	train_ans_df = pd.read_csv('./train.csv')['Survived']
	test_df = pd.read_csv('./test.csv')[features]
	test_ans_df = pd.read_csv('./gender_submission.csv')

	#prep dataframes
	encoder = ce.one_hot.OneHotEncoder()
	train_df = encoder.fit_transform(train_df)
	test_df = encoder.transform(test_df)
	
	#train model
	classifier=KNeighborsClassifier()
	classifier.fit(train_df,train_ans_df)
	predictions=classifier.predict(test_df)

	#test model
	truth = test_ans_df['Survived'].to_numpy()
	print(accuracy_score(truth,predictions), file=open("output.txt", "a"))

print("Based on Class:", file=open("output.txt", "a"))
run_ml(['Pclass'])
print("Based on Sex:", file=open("output.txt", "a"))
run_ml(['Sex'])
print("Based on Class and Sex:", file=open("output.txt", "a"))
run_ml(['Pclass','Sex'])



