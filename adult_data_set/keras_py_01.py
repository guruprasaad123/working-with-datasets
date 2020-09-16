
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Load libraries
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# import csv's
train_data = pd.read_csv('./data/train_preprocessed.csv')
test_data = pd.read_csv('./data/test_preprocessed.csv')

# store target columns in different variable for later usage
train_target = train_data['target']
test_target = test_data['target']

# drop 'target' column 
train_data.drop(['target'],axis=1,inplace=True)
test_data.drop(['target'],axis=1,inplace=True)

# Extracting features
X_train = train_data.values
X_test = test_data.values

# extracting targets
# Y_train = train_target.values.reshape(-1,1)
# Y_test = test_target.values.reshape(-1,1)


# to array 

Y_train = train_target.values
Y_test =  test_target.values

# print('Y shapes ',Y_train_arr.shape , Y_test_arr.shape)

# one-hot encoding
# oneHot.fit(Y_train)
# Y_train_hot = oneHot.transform(Y_train).toarray()

# oneHot.fit(Y_test)
# Y_test_hot = oneHot.transform(Y_test).toarray()

# extracting rows , columns
(m,n) = X_train.shape


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = Sequential()

classifier.add(Dense(6 ,  kernel_initializer='random_normal', bias_initializer='zeros',  activation = 'relu', input_dim = n))
classifier.add(Dense(6,  kernel_initializer='random_normal', bias_initializer='zeros', activation = 'relu'))
classifier.add(Dense(1,  kernel_initializer='random_normal', bias_initializer='zeros', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# This callback will stop the training when there is no improvement in  
# the validation loss for three consecutive epochs. 

history = classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100,callbacks=[callback])

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#classification report
print ("\nClassification Report\n")
print (classification_report(Y_test, y_pred))
print('\nAccuracy Report\n')
print(accuracy_score(Y_test,y_pred))

'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      9763
           1       0.00      0.00      0.00         6

    accuracy                           1.00      9769
   macro avg       0.50      0.50      0.50      9769
weighted avg       1.00      1.00      1.00      9769


Accuracy Report

0.9993858122632818
'''