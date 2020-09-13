import numpy as np 
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

# log dir for tensorboard
def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z
    
def logistic_classifiction(n_inputs,n_hidden1,n_hidden2,n_outputs,learning_rate=0.001):
    with tf.device(device_name):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")
        with tf.name_scope("dnn"):
            hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
            hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
            logits = neuron_layer(hidden2, n_outputs, name="outputs")
        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(xentropy, name="loss")
            loss_summary = tf.summary.scalar('loss_summary',loss)
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            training_op = optimizer.minimize(loss)
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            accuracy_summary = tf.summary.scalar('accuracy_summary',accuracy)
        with tf.name_scope("saver"):
            saver = tf.train.Saver()
        with tf.name_scope("init"):
            init = tf.global_variables_initializer()
    return init , saver , accuracy , accuracy_summary , loss , loss_summary , training_op
    

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
Y_train = train_target.values.reshape(-1,1)
Y_test = test_target.values.reshape(-1,1)


# to array 

Y_train_arr = train_target.values
Y_test_arr =  test_target.values

print('Y shapes ',Y_train_arr.shape , Y_test_arr.shape)

# one-hot encoding
oneHot.fit(Y_train)
Y_train_hot = oneHot.transform(Y_train).toarray()

oneHot.fit(Y_test)
Y_test_hot = oneHot.transform(Y_test).toarray()

# extracting rows , columns
(m,n) = X_train.shape

# hyper parameters
n_epochs = 1001
batch_size = 250
n_batches = int(np.ceil(m / batch_size))
learning_rate = 0.01


n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

init , saver , accuracy , accuracy_summary , loss , loss_summary , training_op = logistic_classifiction(n,n_hidden1,n_hidden2,n_outputs,learning_rate)

logdir = log_dir('logs')

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph() )

checkpoint_path = "/tmp/mylogreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./mylogreg_model"

startTime = datetime.now()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, Y_train_arr, batch_size):
            sess.run(training_op, feed_dict={'X:0': X_batch, 'y:0': y_batch})
        acc_batch = accuracy.eval(feed_dict={'X:0': X_test, 'y:0': Y_test_arr})
        acc_val = accuracy.eval(feed_dict={'X:0': X_test, 'y:0': Y_test_arr })
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")











