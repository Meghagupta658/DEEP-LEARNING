#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:12:06 2021

@author: meghagupta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_digits      #predefined dataset in lib have digits stored in it from 0 to 9
digits = load_digits()
plt.matshow(digits.images[99])    #will give the image of digit how computer reconize it 

plt.matshow(digits.images[77]) 

#for all images from 0 to 9
for i in range(10):
    plt.matshow(digits.images[i])
    
#matrix for each digit image
digits.images[0]    

df = pd.DataFrame(digits.data,columns = digits.feature_names)
df.head()   

df['target'] = digits.target

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.33, random_state = 0)

dt= DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

accuracy_score(y_test,y_pred)

rf = RandomForestClassifier(n_estimators = 90)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)