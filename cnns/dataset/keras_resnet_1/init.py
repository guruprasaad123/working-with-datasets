import numpy as np
import pandas as pd
import os
# for reading and displaying images
from skimage.io import imread
from skimage.transform import resize
# for reading the csv
import csv
# for creating datasets
import h5py
from matplotlib import pyplot as plt

# from utils import *
from model import *

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# import training dataset
train_dataset_path = os.path.join( '../','train.h5')

# reading the contents from the train.h5 file
train_dataset = h5py.File(train_dataset_path,'r')

# training data features
train_x = train_dataset['train_x'][:]
train_y = train_dataset['train_y'][:]
list_class = train_dataset['list_class'][:]

# wrap the values with numpy array
train_x = np.array(train_x)
train_y = np.array(train_y)
list_class = np.array(list_class)

train_y_hot = convert_to_one_hot( train_y , len(list_class) ).T

print('train_x ' , train_x.shape)
print('train_y ' , train_y_hot.shape)
print('list_class ',list_class)


# import testing dataset
test_dataset_path = os.path.join( '../','test.h5')

# reading the contents from the test.h5 file
test_dataset = h5py.File(test_dataset_path,'r')

# training data features
test_x = test_dataset['test_x'][:]
list_class = test_dataset['list_class'][:]

# wrap the values with numpy array
test_x = np.array(test_x)
list_class = np.array(list_class)


array_list = list_class.tolist()

print('test_x ',test_x.shape)
print('list_class ',array_list)
