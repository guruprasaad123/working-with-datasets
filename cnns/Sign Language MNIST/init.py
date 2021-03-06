import pandas as pd
import os
from utils import *

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

path_train_data = os.path.join('sign_mnist_train','sign_mnist_train.csv')

path_test_data = os.path.join('sign_mnist_test','sign_mnist_test.csv')

training_data = pd.read_csv(path_train_data)

testing_data  = pd.read_csv(path_test_data)

# Using the operator .iloc[]  
# to select multiple rows with 
# some particular columns 
# result = df.iloc[ [2, 3, 5], [0, 1] ] 

# Using the operator .iloc[] 
# to select all the rows with 
# some particular columns 
# result = df.iloc[:, [0, 1]] 

# retrieve all rows , from 1 to end ( ignoring 0th column )
train_img_data  = training_data.iloc[ :,1:].values

# reshaping the 784 into 28 x 28 x 1 ( H x W X C )
train_img_data = train_img_data.reshape((-1,28,28,1))

# retrieve all rows , only 0 th column 
train_img_label = training_data.iloc[ :,0 ].values

train_img_label = train_img_label.reshape((1,-1))

# retrieve all rows , from 1 to end ( ignoring 0th column )
test_img_data = testing_data.iloc[:,1:].values

# reshaping the 784 into 28 x 28 x 1 ( H x W X C )
test_img_data = test_img_data.reshape((-1,28,28,1))

# retrieve all rows , only 0 th column 
test_img_label = testing_data.iloc[ :,0 ].values

test_img_label = test_img_label.reshape((1,-1))

classes = np.unique(test_img_label)

train_y = convert_to_one_hot(train_img_label , len(classes) + 1 ).T

test_y = convert_to_one_hot(test_img_label , len(classes) + 1 ).T

print('Classes : {}'.format( len(classes) ) )

print('training image data',train_img_data.shape)
print('train_label_data',train_img_label.shape)

print('testing image data',test_img_data.shape)
print('test_label_data',test_img_label.shape)

print('Training Label : ',train_y.shape)

print('Test Label : ',test_y.shape)


_, _, parameters = model(train_img_data, train_y , test_img_data , test_y)