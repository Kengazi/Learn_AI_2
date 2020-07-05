#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:02:30 2020

@author: kendalljohnson
"""

"""
***** first pip3 install scipy **************
***** Second pip3 install scikit-learn ******

Week 3 - A base in using data science with python

3.1 :: Using Titanic data set for Machine Learning

The goal of this assignment is to get you comfortable with real datasets and linear regression models.

Scikit-learn is an amazing tool for machine learning that provides models in a few lines

Linear Regression Youtube video link: https://www.youtube.com/watch?v=8jazNUpO3lQ :: Watch it

"""
# Title 
print('A base in using data science with python using scikit learn(sklearn) - guide')

# Imports
import numpy as np
np.random.seed(18) # controls randomization not needed but used for repeatition
import matplotlib.pyplot as plt
import pandas as pd
# Sci-kit learn model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

"""
This Data set is on the 887 of the 3000+ people from the titanic sinking
it contains information like there name, the class they were in, gender, age, number of siblings abort,
number of parents / grand parents abort, and if they survived the sinking

"""


# Data import
df = pd.read_csv('titanic.csv')

# Data from titanic csv
lived = df.values[:,0]              # 1 is survived 0 did not
clas = df.values[:,1]               # 1st class, 2nd, or 3rd
name = df.values[:,2]               # name of passenger "string"
gender = df.values[:,3]             # gender of passenger  "string"
age = df.values[:,4]                # age of passenger 
sib = df.values[:,5]                # number of brothers and sisters aboard
parent = df.values[:,6]             # number of parents / grandparents aboard
fare= df.values[:,7]                # how much their ticket cost


# Turning it into a labeled dataframe
    
lived = np.array(lived)
lived_df = pd.DataFrame({"lived":lived}) 

clas = np.array(clas)
clas_df = pd.DataFrame({"clas":clas})

name = np.array(name)
name_df = pd.DataFrame({"name":name})

gender = np.array(gender)
gender_df = pd.DataFrame({"gender":gender})

age = np.array(age)
age_df = pd.DataFrame({"age":age})

sib = np.array(sib)
sib_df = pd.DataFrame({"sib":sib})

parent = np.array(parent)
parent_df = pd.DataFrame({"parent":parent})

fare = np.array(fare)
fare_df = pd.DataFrame({"fare":fare})

"""
The training data is used to train the model to give it the ability to make assumetions of the test data
"""

# Seperate data into training and testing data using train_test_split making 80% of the data training and 20% test data
X_train, X_test, y_train, y_test = train_test_split(sib, parent, test_size = 0.2)

train_size = len(X_train)
test_size = len(X_test)

# Reshape from (404,) to (404,1)
X_train = X_train.reshape(train_size,1)
y_train = y_train.reshape(train_size,1)
X_test = X_test.reshape(test_size,1)
y_test = y_test.reshape(test_size,1)


# Lin Regression Model :: 

# Creating an object from import for use
model = LinearRegression()

# Fitting the data in the model *** Important part ***
model.fit(X_train, y_train)

# Model Score *** Important value *** 
m = model.score(X_test,y_test) * 100 # The closer the model score is to 100 the better
print("The Model Score is {:.4} out of 100%".format(m))

# Using the model to preditions
y_pred = model.predict(X_test)

# R squared :: determiner of how good the prediction fits
r2 = r2_score(y_test,y_pred) 
print("The R2 Score is {:.4} out of 1 ".format(r2))

# Plotting 
plt.scatter(sib,parent,color = 'red',label = "Data",marker = '.')

# Line of Best Fit based on model using Linear Model
plt.plot(X_test,y_pred,color = 'blue',label = 'Best fit')  

# Rest of Plotting features
plt.xlabel('number of siblings aboard')
plt.ylabel("number of parents aboard")                  # Also includes grand-parent so can have up to 6
plt.grid()                                              # Plot grid lines
plt.legend()                                            # Plot legend
plt.title("Number of siblings vs number of parents aboard")
plt.minorticks_on()                                     # adds small ticks on main axis

# Create Excel file

# Putting frames together 

frame = [lived_df,clas_df, name_df, gender_df, age_df, sib_df, parent_df, fare_df]
result = pd.concat(frame, axis=1)

# Save a excel file :: You need to change this to you saving areas 

result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Titanic.xlsx')

"""
Your turn...

Its easy to see why the Line of Best fit has a positive slope because more brothers and sister abord the more likely you are younger
and therefore have more parents aboard.

What can you say about this model with a model score of 32/100 and 
the R2 value which is .32 (and should be 1-ish)?

Is this a good model?

Is this data well represented?


Create a model, graph, and a line of best fit for x:age vs y:number of siblings 

print model and r2 score

Explain results and opinion about graph and why it is that way? same as above


Create a model, graph, and a line of best fit for x:age vs y:fare  

print model and r2 score

Explain results and opinion about graph and why it is that way? same as above

"""
