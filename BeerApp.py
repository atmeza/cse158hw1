#!/usr/bin/env python
"""
Filename: BeerApp.py
Author: Alex Meza
Date: 10/1/17
Description:
	...
"""

import pandas as pd
import numpy as np
import math
from collections import Counter, defaultdict
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



def main():
	#read in file of beer survey's to panda database

	fname = "./beer_50000.json"

	#used pickle to save time and not have to upload file into datafarme
	
	#original command to create pickle
	#pd.DataFrame(parseData(fname)).to_pickle("./beer.pkl")
	
	#read dataframe from pickle(SERIOUSLY SO MUCH FASTER)
	data = pd.read_pickle("./beer.pkl")
	
	#fill in na values with a number 
	data.fillna(-99999, inplace = True)
	#print out variance and mse using mean as predicted value of reviewtaste column
	reviewtaste = data['review/taste']
	print("Variance of review/taste ratings: ",variance(reviewtaste))
	print("MSE of review/taste ratings: ",MSE(reviewtaste,np.full((len(reviewtaste),1),np.mean(reviewtaste))))
	
	#print out average taste rating per style
	beerAverages(data)

	#use linear regression to predict tast rating of american ipa
	predict_ReviewTaste(data)
	predictIfAmericanIPA(data)
	
	



#open file and parse data from file line by line

def parseData(fname):
	with open(fname) as fname:
		for l in fname: 
    			yield eval(l)

#find variance of given data
def variance(data):
	mean = np.mean(data)
	var = 0
	for rating in data:
		var += (rating - mean)**2

	return var/len(data)

#find mse given data and prediction that is the mean
def MSE(data, predict):
	mse = 0
	for rating, prediction in zip(data, predict):
		mse +=(rating - prediction)**2

	return (mse/len(data))


#print out all styles of beer and their averages
def beerAverages(data):
	#create dictionary of lists 
	style_taste = defaultdict(list)
	
	#for each beer style and rating record in style_taste dictionary
	for indices, row in data.iterrows():
		style = row['beer/style']
		rating = row['review/taste']
		style_taste[style].append(rating)
	#print how many beers
	print("There are ", len(style_taste), " styles of beer")

	#comput average beer rating per style and printout
	for style in style_taste:
		average = 0
		for rating in style_taste[style]:
			average+= float(rating)
		styleaverage = average/len(style_taste[style])
		print("Style: ",style, " Average taste review: ", styleaverage)


#questions 3-5
def predict_ReviewTaste(data):

	#create feature matrix
	X = [ featureAmericanIPA(row['beer/style']) for d, row in data.iterrows()  ]
	X = np.column_stack((np.full((len(X),1),1), X))
	
	#create label
	y = data['review/taste']

	#run lin regression on features and labels
	theta,residuals,rank,s = np.linalg.lstsq(X, y)
	print("theta[0] gives predicted rating for non American IPA")
	print("Predicted taste rating for non American IPA: ",theta[0])
	print("theta[1], ", theta[1], " gives difference between rating of American IPA's and others")
	print("Predicted taste rating for American IPA: ",(theta[0]+theta[1]))

	#cross validation
	Xtrain = X[:int(len(X)/2)]
	Xtest = X[int(len(X)/2):]
	ytrain = y[:int(len(y)/2)]
	ytest = y[int(len(y)/2):]
	
	#find lin regression and print out mse of training and testing set
	print("After Cross Validation feature if is American IPA")
	theta, residuals, rank, s = np.linalg.lstsq(Xtrain, ytrain)
	print("MSE of training set: ", MSE(ytrain, Xtrain.dot(theta)))
	print("MSE of test set: ", MSE(ytest, Xtest.dot(theta)))

	print("After Cross Validation and features are all different styles of beers")
	#create dictionary of lists 
	style_taste = []
	
	#for each beer style and rating record in style_taste dictionary
	for indices, row in data.iterrows():
		style = row['beer/style']
		if style not in style_taste:
			style_taste.append(style)

	#create feature matrix for all different styles of beer and perform cross validation
	X = [featureAllStyles(row['beer/style'], style_taste) for d, row in data.iterrows()] 	
	X = np.array(X)
	Xtrain = X[:int(len(X)/2)]
	Xtest = X[int(len(X)/2):]
	ytrain = y[:int(len(y)/2)]
	ytest = y[int(len(y)/2):]
	
	theta, residuals, rank, s = np.linalg.lstsq(Xtrain, ytrain)
	print("Theta: ",theta)
	print("MSE of training set: ", MSE(ytrain, Xtrain.dot(theta)))
	print("MSE of test set: ", MSE(ytest, Xtest.dot(theta)))


	return theta

def featureAllStyles(style, styles):
	row = []
	for beer_type in styles:
		if(style == beer_type):
			row.append(1)
		else:
			row.append(0)
	return row
	

#return binary value of whether or not style is an american ipa
def featureAmericanIPA(d):
	
	if(d == 'American IPA'):
		return 1
	return 0	



#questions 6-8
def predictIfAmericanIPA(data):
	beer_ABV = data['beer/ABV']
	review_taste = data['review/taste']
	X = np.column_stack((beer_ABV, review_taste))
	y = [featureAmericanIPA(d) for d in data['beer/style']]
	Xtrain = X[:int(len(X)/2)]
	Xtest = X[int(len(X)/2):]
	ytrain = y[:int(len(y)/2)]
	ytest = y[int(len(y)/2):]
	
	clf = svm.LinearSVC(C=1000)
	clf.fit(Xtrain, ytrain)
	print("Prediction of whether or not beer is an American IPA based on ABV/taste")
	print("svm.LinearSCV predictions with C=1000")
	print("Training accuracy: ", clf.score(Xtrain, ytrain))
	print("Testing Accuracy: ",clf.score(Xtest, ytest))
	
	print("Predictions of whether beer is an IPA based on ABV/taste/overallreview/aroma")
	print("data scaled, and shuffled")
	X = np.column_stack((X, data['review/overall'],data['review/aroma']))
	X = preprocessing.scale(X)

	Xtrain, Xtest, ytrain, ytest = cross_validation.train_test_split( X, y, test_size = .5)	

	clf1 = svm.LinearSVC(C=.1)
	clf1.fit(Xtrain,ytrain)
	print("svm.LinearSCV predictions with C=.1")
	print("Training accuracy: ", clf1.score(Xtrain, ytrain))
	print("Testing Accuracy: ",clf1.score(Xtest, ytest))
	

	clf2 = svm.LinearSVC(C=10)
	clf2.fit(Xtrain,ytrain)
	print("svm.LinearSCV predictions with C=10")
	print("Training accuracy: ", clf2.score(Xtrain, ytrain))
	print("Testing Accuracy: ",clf2.score(Xtest, ytest))
	
	clf3 = svm.LinearSVC(C=1000)
	clf3.fit(Xtrain,ytrain)
	print("svm.LinearSCV predictions with C=1000")
	print("Training accuracy: ", clf3.score(Xtrain, ytrain))
	print("Testing Accuracy: ",clf3.score(Xtest, ytest))
	


	clf4 = svm.LinearSVC(C=100000)
	clf4.fit(Xtrain,ytrain)
	print("svm.LinearSCV predictions with C=100000")
	print("Training accuracy: ", clf4.score(Xtrain, ytrain))
	print("Testing Accuracy: ",clf4.score(Xtest, ytest))
	
	print("Changing the regularizer C affects the accuracy of the model\n changes accuracy at a quadratic rate")
	print("but eventually converges to a relaxed state\n small numbers increases accuracy and as C grows accuracy decreases")



if __name__ =="__main__":
	main()

