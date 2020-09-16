import numpy as np 
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

oneHot = OneHotEncoder() 

device_name = 'cpu' if sys.argv[1] else 'gpu'  # Choose device from cmd line. Options: gpu or cpu

if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

# to get Random N rows X , Y batches
def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

# log dir for tensorboard
def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

def logistic_classifiction(cols,learning_rate=0.001):
    with tf.device(device_name):
        X = tf.placeholder( tf.float32, [None,cols] ,name="X")
        Y = tf.placeholder( tf.float32 ,[None,1] ,name="Y")
        with tf.name_scope('logistic_regression'):
            with tf.name_scope('train'):
                
                initializer = tf.random_uniform([cols,1],-1.0,1.0)
                W = tf.Variable(initializer ,name='W')
                b = tf.Variable(tf.zeros([1]) ,name="b")
                logits = tf.add(tf.matmul(X,W),b)
                y_proba = tf.sigmoid(logits)

                loss = tf.losses.log_loss(Y,y_proba,scope='loss')

                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

                training_op = optimizer.minimize(loss)

                loss_summary = tf.summary.scalar('log_loss', loss)


            with tf.name_scope('save'):
                saver = tf.train.Saver();

            with tf.name_scope('init'):
                # for global initialization
                init = tf.global_variables_initializer()

    return init , y_proba , saver , loss , training_op , loss_summary 
    
    


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

# Using Standard Scaler
sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)

# extracting targets
Y_train = train_target.values.reshape(-1,1)
Y_test = test_target.values.reshape(-1,1)

print('Y shapes ',Y_train.shape , Y_test.shape)

# one-hot encoding
oneHot.fit(Y_train)
Y_train_hot = oneHot.transform(Y_train).toarray()

oneHot.fit(Y_test)
Y_test_hot = oneHot.transform(Y_test).toarray()

# extracting rows , columns
(m,n) = X_train.shape

# hyper parameters
n_epochs = 20001
batch_size = 250
n_batches = int(np.ceil(m / batch_size))
learning_rate = 0.01

init , y_proba , saver , loss , training_op , loss_summary = logistic_classifiction(n,learning_rate)

logdir = log_dir('logs')

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph() )

checkpoint_path = "/tmp/mylogreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./mylogreg_model"

startTime = datetime.now()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    
    if os.path.isfile(checkpoint_epoch_path):
        
        # if the checkpoint file exists, restore the model and load the epoch number
        
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        
        print("Training was interrupted. Continuing at epoch", start_epoch)

        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, Y_train, batch_size)
            sess.run(training_op, feed_dict={'X:0': X_batch, 'Y:0' : y_batch})
        loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={ 'X:0' : X_test , 'Y:0' : Y_test })
        file_writer.add_summary(summary_str, epoch)
        if epoch % 500 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))

    saver.save(sess, final_model_path)
    y_proba_val = y_proba.eval(feed_dict={ 'X:0' : X_test, 'Y:0' : Y_test})
    os.remove(checkpoint_epoch_path)

    y_pred = (y_proba_val >= 0.5)

    p_score = precision_score(Y_test, y_pred)

    r_score = recall_score(Y_test, y_pred)

    acc_score = accuracy_score(Y_test , y_pred)

    print('Precision Score = {} , Recall Score = {} , Accuracy = {}'.format(p_score,r_score,acc_score))

    endTime = datetime.now()

    print('time took ',endTime - startTime)













