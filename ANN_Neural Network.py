#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:30:51 2021

@author: meghagupta
"""

import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data)
df.columns = iris.feature_names
df.head()
iris.target_names
df['target'] = iris.target

feature = iris.data                    #input
targets = iris.target.reshape(-1,1)    #output : target variable

from sklearn.preprocessing import OneHotEncoder    #OneHotEnoder is used to convert the data categorical data into numerical data
ohe = OneHotEncoder()
targets = ohe.fit_transform(targets).toarray()

#split data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, targets, test_size = 0.3, random_state = 0)

#keras
import keras

from keras.models import Sequential
from keras.layers import Dense         #dense for creating dense network
from keras.optimizers import Adam      #Adam is an optimizer we choosed for adjusting weight through Gradient Descent

#creating a simple neural network
model = Sequential()   #Sequential is ANN model it creates the artificial neural network, with input and ouput layer but not hidden layers
#hidden layer is created
model.add(Dense(10, input_dim = 4, activation='sigmoid'))    #this will add the hidden layer that is dense neural network with 10 neurons in the hidden layers and input_dim = 4 because there are 4 inputs
#output layer
model.add(Dense(3,activation ='softmax'))    #dense is 3 as there are 3 varaiables in target that is 3 outputs

optimizer = Adam()

model.compile(loss='categorical_crossentropy',optimizer=optimizer , metrics = ['accuracy'])
model.fit(X_train,y_train,epochs=50,verbose=2)       #epochs=50 means it will run for 50 iterations

model.evaluate(X_test,y_test)
