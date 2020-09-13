import numpy as np 
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder 
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import os

oneHot = OneHotEncoder() 

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
    X = tf.placeholder( tf.float32, [None,cols] ,name="X")
    Y = tf.placeholder( tf.float32 ,[None,2] ,name="Y")
    with tf.name_scope('logistic_regression'):

        with tf.name_scope('train'):
            W = tf.Variable( tf.zeros( [cols,2] ) )
            # Trainable Variable Bias 
            b = tf.Variable(tf.zeros([2])) 
            # hypothesis
            Y_hat = tf.nn.sigmoid( tf.add( tf.matmul(X,W) , b ) )
            # cost
            cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( labels=Y , logits=Y_hat ) )
            # cost summary
            cost_summary = tf.summary.scalar('cost_summary',cost)
            # optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # training operation
            training_op = optimizer.minimize(cost)

        with tf.name_scope('eval'):
            # accuracy
            correct_prediction = tf.equal( tf.argmax(Y_hat,1) , tf.argmax(Y,1) )
            accuracy = tf.reduce_mean( tf.cast( correct_prediction,tf.float32 ) )
            accuracy_summary = tf.summary.scalar('accuracy_summary',accuracy)

        with tf.name_scope('save'):
            saver = tf.train.Saver();

        with tf.name_scope('init'):
            # for global initialization
            init = tf.global_variables_initializer()

    return init , saver , training_op , X , Y_hat , cost , cost_summary , accuracy , accuracy_summary



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

print('Y shapes ',Y_train.shape , Y_test.shape)

# one-hot encoding
oneHot.fit(Y_train)
Y_train_hot = oneHot.transform(Y_train).toarray()

oneHot.fit(Y_test)
Y_test_hot = oneHot.transform(Y_test).toarray()

# extracting rows , columns
(m,n) = X_train.shape

# hyper parameters
n_epochs = 1000
batch_size = 250
n_batches = int(np.ceil(m / batch_size))
learning_rate = 0.001

init , saver , training_op , X , Y_hat , cost , cost_summary , accuracy , accuracy_summary = logistic_classifiction(n,learning_rate)

logdir = log_dir('logs')

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph() )

checkpoint_path = "/tmp/my_logcls_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_logcls_model"

with tf.Session() as sess:

    if os.path.isfile(checkpoint_epoch_path):
        # if the checkpoint file exists, restore the model and load the epoch number
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    sess.run(init)

    print( 'X name ',X.op.name)

    for epoch in range(start_epoch,n_epochs):

        for batch in range(n_batches):

            X_batch  , y_batch = random_batch(X_train,Y_train_hot,batch_size)

            sess.run( training_op  , 
            feed_dict={ 
            'X:0' : X_batch , 
            'Y:0' : y_batch
            })

            Y_val , cost_val , cost_summary_val , acc_val , acc_sum_val = sess.run( [ 
                Y_hat , 
                cost , 
                cost_summary , 
                accuracy , 
                accuracy_summary 
                ] , 
                feed_dict={ 
                'X:0' : X_test , 
                'Y:0' : Y_test_hot 
                } )

            step = epoch * n_batches + batch

            file_writer.add_summary(cost_summary_val,step)

            file_writer.add_summary(acc_sum_val,step)

        saver.save(sess,final_model_path)

        with open(checkpoint_epoch_path,'wb') as f:
            f.write(b"%d" % (epoch + 1))

        if epoch % 100 == 0 :

            print('at epoch : {} , batch : {} => loss = {} , accuracy = {} '.format(epoch,batch,cost_val,acc_val))

    Y_prob = Y_val

print('after training Y = ',Y_prob)














