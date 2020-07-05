#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:13:40 2020

@author: kendalljohnson
"""

print('Using K-Fold \n')

"""
Week 6 - A base in using data science with python

6.1 :: Using Scikit-learn (sklearn) for Machine Learning

The goal of this assignment is to get you comfortable with analysis on ML model to question them a show their true quility.

K-fold is basically finding the average model score of a data set with for choosen model, but it cycles through different
combos of test data and training data

Scikit-learn is an amazing tool for machine learning that provides datasets and models in a few lines

K-fold Youtube video link: https://www.youtube.com/watch?v=gJo0uNL-5Qw&t=1086s :: Watch it
"""

# Title 
print('A base in using data science with python using pandas - guide')

# Imports :: modules used to create code

import pandas as pd                               # DataFrame Tool
import numpy as np                                # Basically a great calculator for now
from sklearn import linear_model                  # For our Linear Regression Model // Sklearn is a M.L. modules
from sklearn.svm import SVC
"""
This Data set is on housing information from Washington houses

"""

# Data import (the xlsx file has names attached)

df = pd.read_csv('Housing_Data.csv') 

# Data from titanic xlsx
numbers = df.values[:,0] 
price = df.values[:,1]              # 1 price of house in 1000s
bed = df.values[:,2]               # number of bedrooms
bath = df.values[:,3]               # number of bathrooms
square_ft = df.values[:,4]             # square feet of house
lot_ft = df.values[:,5]                # square feet of house + land
floors = df.values[:,6]                # number of floors
veiw = df.values[:,7]             # number of parents / grandparents aboard

num = len(numbers)

# Definition 

def Score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    S = model.score(X_test,y_test)*100
    return S

# Turning it into a dataframe
    
price = np.array(price)
price_df = pd.DataFrame(price) 

bed = np.array(bed)
bed_df = pd.DataFrame(bed)

bath = np.array(bath)
bath_df = pd.DataFrame(bath)

square_ft = np.array(square_ft)
square_ft_df = pd.DataFrame(square_ft)

lot_ft = np.array(lot_ft)
lot_ft_df = pd.DataFrame(lot_ft)

floors = np.array(floors)
floors_df = pd.DataFrame(floors)

veiw = np.array(veiw)
veiw_df = pd.DataFrame(veiw)


# Using Linear Regression
model = linear_model.LinearRegression()

# Using KFold

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

# K = 3
kf = KFold(n_splits = 3)

# Another Kfold method
KS  = StratifiedKFold(n_splits = 3)

# Blank list to put model scores
models = []

# A for list that moves the data in training and test data. Then runs the model and save the model scores.
for train_index, test_index in kf.split(numbers):
    X_train, X_test,y_train, y_test = square_ft[train_index],square_ft[test_index],price[train_index],price[test_index]
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')
    train_size = len(train_index)
    test_size = len(test_index)
    
    # Reshape from (num,) to (num,1)
    X_train = X_train.reshape(train_size,1)
    #y_train = y_train.reshape(train_size,1)
    X_test = X_test.reshape(test_size,1)
    #y_test = y_test.reshape(test_size,1)
    
    # Scores
    S = Score(model,X_train, X_test,y_train, y_test)
    models.append(S)
    
# Get average of the incoming scores
avg = np.mean(models)
 
print("The average model score using KFold is {:}%".format(avg))   

"""
Your Turn

Change the n_splits in KFold.. What changes / improvements do you see?

Find the average model score for Logistic, SVM, and Randomforest for age

Find the average model score for Logistic, SVM, and Randomforest for gender

Find the average model score for Logistic, SVM, and Randomforest for fare

# BONUS use the StratifiedKFold method for these same data


"""