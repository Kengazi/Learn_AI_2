#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:07:10 2020

@author: kendalljohnson
"""

# Naive Bayes
"""
Week 5 - A base in using data science with python

5.2 :: Using Scikit-learn (sklearn) for Machine Learning

The goal of this assignment is to get you comfortable with doing working with ML models like Naive Bayes(NB).

Naive Bayes : Much more classical statistical approach to ML using Guassians and Probabilites.

Scikit-learn is an amazing tool for machine learning that provides datasets and models in a few lines

Naive Bayes Youtube video link: https://www.youtube.com/watch?v=PPeaRc-r1OI :: Watch it

"""

# Title 
print('A base in using data science with python using pandas - guide')

# Imports :: 
import numpy as np
import pandas as pd

# ML
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
np.random.seed(1)
"""
This data is based on a made up statistics of variables that go into a student graduating college in 4 years
The varables are goes to class 90% of the time, studys 3 hours+ a day, Family visits often, has more then 3 close college friends,
GPA over a 2.0, and if that did graduate ontime
"""

# Data  :: Binary Data set on factors of graduations on students

df = pd.read_excel('Grad_data.xlsx')     # Graduation data
Class9 = df.values[:,0]                  # Was present in class at least 90%
Study3 = df.values[:,1]                  # Studied atleast 3 hours a weekday
FamilyV = df.values[:,2]                 # Family visited often
Friends3	 = df.values[:,3]                # Had more then 3 close friends
GPA2 = df.values[:,4]                    # Gpa over 2.0
Grad = df.values[:,5]                    # Graduated on time

# Create DataFrame
Class9 = pd.DataFrame(Class9)
Study3 = pd.DataFrame(Study3)
FamilyV = pd.DataFrame(FamilyV)
Friends3 = pd.DataFrame(Friends3)
GPA2 = pd.DataFrame(GPA2)
Grad = pd.DataFrame(Grad)

# Putting pandas DataFrames together 
frames = [Class9,Study3,FamilyV,Friends3,GPA2]
inputs = pd.concat(frames, axis=1)

# Machine Learning :: Naive Bayes

# Splitting Training and Test data
X_train, X_test,y_train, y_test = train_test_split(inputs,df.Grad,test_size = 0.2)

# Naive Bayes :: the Gaussian and the Bernoulli both very statistical models
model = GaussianNB()
#model = BernoulliNB()

# Fitting the model
model.fit(X_train,y_train)

# The model score
m = model.score(X_test,y_test) * 100
print("The model score is {:.4}%".format(m))

# Predicting Data
Pred = model.predict([[0,1,1,0,1]])
Pred = Pred[0]
if Pred == 1:
    print('Graduated ontime')
else:
    print('Did not Graduated ontime')
    
"""
Your turn

Remove the np.random.seed() on line 36

Use the model / BernoulliNB() How does your model score change?

In the Excel File named Grad2_data.xlsx

there is this data plus 2 extra catagories GForBF which is if the indivual had a girlfriend or boyfriend for most
of college, and if the student live on or of campus

Use this excel file to make preditions on the graduation status of these students 

GForBF = df.values[:,5]                    
Campus = df.values[:,6] 
Grad =  df.values[:,7]  


Paul = [0,1,0,0,0,1,1]

Beth = [1,1,0,0,0,0,1]

TJ = [0,0,0,1,0,1,1]

Rebecca = [1,1,1,1,0,1,0]

Ngazi = [1,1,0,1,0,1,1]

What is you opinion of this data comparied to Decision Tree and SVM? 

"""