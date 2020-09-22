import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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
    
    


dataframe = pd.read_csv('../data/adult_preprocessed.csv')

X = dataframe.iloc[:, 0:88].values
y = dataframe.iloc[:, 88].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# extracting rows , columns
(m,n) = X_train.shape

# hyper parameters
n_epochs = 20001
batch_size = 250
n_batches = int(np.ceil(m / batch_size))
learning_rate = 0.001

y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)

print('y ',y_test.shape , y_train.shape)

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
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={'X:0': X_batch, 'Y:0' : y_batch})
        loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={ 'X:0' : X_test , 'Y:0' : y_test })
        file_writer.add_summary(summary_str, epoch)
        if epoch % 500 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))

    saver.save(sess, final_model_path)
    y_proba_val = y_proba.eval(feed_dict={ 'X:0' : X_test, 'Y:0' : y_test})
    os.remove(checkpoint_epoch_path)

    y_pred = (y_proba_val >= 0.5)

    p_score = precision_score(y_test, y_pred)

    r_score = recall_score(y_test, y_pred)

    f1score = f1_score(y_test,y_pred)

    accuracy_scr = accuracy_score(y_test,y_pred)

    confustionmatrix = confusion_matrix(y_test,y_pred)

    print('Precision Score = {} , Recall Score = {} , f1 Score = {} \n'.format(p_score,r_score,f1score))

    print('Confusion Matrix = {} \n'.format(confustionmatrix))

    class_report = classification_report(y_test,y_pred)

    print('Accuracy = {}\n'.format(accuracy_scr) )

    print(class_report)

    endTime = datetime.now()

    print('time took ',endTime - startTime)













