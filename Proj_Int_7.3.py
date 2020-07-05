#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:18:11 2020

@author: kendalljohnson
"""

"Keras ANN"

"""
***** first :sudo pip3 install tensorflow **************
***** first :sudo pip3 install keras **************

Week 7 - A base in using data science with python

7.3 :: Introduction of Artifical Neural Networks for Machine Learning

The goal of this assignment is to get you comfortable with analysis on ML model without scikit learn.

We are done with sklearn models and now will focus on Keras and Pytorch ANNs 

ANN is a step in ML and true AI being it based on a human system. We will only be using this going forward.

ANN keras Youtube video link: https://www.youtube.com/watch?v=XNKeayZW4dY&t=923s :: Watch it

"""

# Imports
import numpy as np
#import matplotlib.pyplot as plt

# The module keras that is used for ANNs
from keras.datasets import mnist

# The module keras that is used for ANNs
from keras.models import Sequential

# The properties for ANNs
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

""" 
This data is an 28x28 images of handwritten digits with changes in pixel intensities
"""

# Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizing
X_test = X_test/ 255
X_train = X_train/ 255

# Changing to categories
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Reshaping to fit in keras neural net
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

# Main Script

# Way layers are added together 
model = Sequential()

# adding  model layers
model.add(Dense(50, activation='relu', input_shape=(28,28,1))) # regular hidden layer with 50 nodes but also take is the input shape
model.add(Dense(50, activation='relu'))# regular hidden layer with 50 nodes
model.add(Flatten()) # changes layes to 1-D
model.add(Dense(10, activation = 'softmax'))# output layer with nodes to match each category

# Model summary that shows the properties of the model structure
# model.summary() # 

# Puts all the model features together 
model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy']) #loss='sparse_categorical_crossentropy'

# Now adding the training data in batch size of 1000
model.fit(X_train,y_train,batch_size = 1000)

# Model Score
m = model.evaluate(X_test,y_test)
m_eval = m[1]
m1 = m_eval*100
print('The model score is {} percent'.format(m1))

# Predicting 
yp = model.predict(X_test)
real = np.argmax(y_test[1])
print('The real is {}'.format(real))
predicted = np.argmax(yp[1])
print('The predicted is {}'.format(predicted))

"""
Your Turn...

My goal for you in this script is not for you to replicate but to understand

This is a more complex way of coding ANNs for more acceracy and precision compared to sklearn 
for bigger problems

# BONUS Change parameters 

Change the Activation functions to softmax
Change the Optimizer
Change the Hidden layers
Change the number of iterrations

Confirm other data being predictable

Does the model improve?

Get over a 95% Model score

"""

"""
Project Mid_point 2
*******************************************************************************
Use your new understanding to create an ANN (PyTorch or Keras) for the same data in part 2 of Project-Midpoint
Get a model score over 85% and send me the script named Proj_Mid_point_2.py
*******************************************************************************

"""