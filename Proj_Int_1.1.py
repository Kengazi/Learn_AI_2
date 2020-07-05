#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 07:56:52 2020

@author: kendalljohnson
"""
"""
Week 1 - A base in using data science with python

1.1 :: Basic Math in python and importing NumPy

The goal of this assignment is to get you comfortable with doing math in python with numpy.

NumPy is by far the most versitile module in python and least computationally expensive.

Numpy Youtube video link: https://www.youtube.com/watch?v=8JfDAm9y_7s :: Not the most nessecary for this week

"""

# Title 
print('A base in using data science with python - guide')

# Imports 
import numpy as np #:: NumPy or number python is a great way easily using mathmatical operations 
# and a common way of importing numpy is np 

# Varables 
a = 1
b = 2
c = 3
d = 4

x = 10
y = 20
z = 30

# Basic fuctions 
print('Basic fuctions\n') # \n creates a spaces a line between the next print line

# numbers as floats
flo = np.float(a)
print('Answer is {}'.format(flo))

# numbers as intergers
inter = np.int(flo)
print('Answer is {}'.format(inter))

# Adding
A = np.add(a,b)
print('Answer is {}'.format(A))

# Subtraction
B = np.subtract(b,a)
print('Answer is {}'.format(B))

# Multiplication
X = np.multiply(x,y)
print('Answer is {}'.format(X))

# Division
Y = np.divide(y,x)
print('Answer is {}\n'.format(Y))


# More Complex Functions
print('Basic exponental, log, and square root fuctions\n')

# e, log, and square root  

ex = np.exp(0) # e^(x)
print('Answer is {}'.format(ex))

log = np.log(x) # ln(x)
print('Answer is {:.4}'.format(log)) # {:.4} rounds the decimal place down to 1000th place ... I would like you to do this  

log10 = np.log10(x) # log10(x)
print('Answer is {}'.format(log10))

square = np.sqrt(d) 
print('Answer is {}\n'.format(square))

# 9 = 3**2 # reg power function

# Trig
print('Trig\n')

# More Varables in radians

# numpy can be used to retrive the number pi
pi = np.pi
pi_2 = pi/2
pi_3 = pi/3
pi_4 = pi/4

# Converting from rad to degrees 
deg = np.degrees(pi)
print('Answer is {}'.format(deg))

# Converting from degrees to rad
rad = np.radians(deg)
print('Answer is {:.4}'.format(rad))

# sine
sin_1 = np.sin(pi_3)
print('Answer is {:.4}'.format(sin_1))

sin_2 = np.sin(pi_4)
print('Answer is {:.4}'.format(sin_2))

# cosine
cos_1 = np.cos(pi)
print('Answer is {:.4}'.format(cos_1))

cos_2 = np.cos(pi_2)
print('Answer is {:.4}'.format(cos_2))

# tangent
tan = np.tan(pi)
print('Answer is {:.4}'.format(tan))

# arcsin
arcsin = np.arcsin(a)
print('Answer is {:.4}'.format(arcsin))

# arccos
arccos = np.arccos(a)
print('Answer is {:.4}'.format(arccos))

# arctan
arctan = np.arctan(pi_3)
print('Answer is {:.4}'.format(arctan))

# hypotenuse of right trigangle with side x and y
hypot = np.hypot(x,y)
print('Answer is {:.4}\n'.format(hypot))


# Important Extras 
print("Important Extras\n")
num = -1.45

# Rounds number down
floor = np.floor(num)            
print('Answer is {}'.format(floor))

# Rounds number up
ceil = np.ceil(num)             
print('Answer is {}'.format(ceil))

# closest interger
trun = np.trunc(num)            
print('Answer is {}'.format(trun))

# Exact copy very useful
copy = np.copy(num)              
print('Answer is {}'.format(copy))

# Finds absolute values
absolute = np.abs(num)
print('Answer is {}'.format(absolute))


"""
 Your turn (Remember to print your answers and round to atleast the thousandth place)

 show secant of pi/2, pi, and 2*pi

 Answer follow equations and find both the float and int

 y = e^(2*x) if x = 2, 4, 7, and 10

 z = log(2xy) if x = 10, 12, and 30; and y = 20, 2 ,and 50

 c = sin(1/3*pi*b) * A^2 if A = 4, 5, and 6; and b = .42, .86, and 1

 d = A*cos(4*a*b*pi) - 12/c if A = 2, a = 4, b = 1.3 and c = 1.25

 Bonus find Surface Area and Volume of a sphere with radius 1.5 meters (dont forget units in this answer)

"""