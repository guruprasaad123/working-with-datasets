import numpy as np
import matplotlib.pyplot as plt
import cifar_tools
import tensorflow as tf

data, labels = cifar_tools.read_data('C:\\Users\\abc\\Desktop\\temp')

x = tf.placeholder(tf.float32, [None, 150 * 150])
y = tf.placeholder(tf.float32, [None, 2])

w1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
b1 = tf.Variable(tf.random_normal([64]))

w2 = tf.Variable(tf.random_normal([5, 5, 64, 64]))
b2 = tf.Variable(tf.random_normal([64]))

w3 = tf.Variable(tf.random_normal([6*6*64, 1024]))
b3 = tf.Variable(tf.random_normal([1024]))

w_out = tf.Variable(tf.random_normal([1024, 2]))
b_out = tf.Variable(tf.random_normal([2]))

def conv_layer(x,w,b):
    conv = tf.nn.conv2d(x,w,strides=[1,1,1,1], padding = 'SAME')
    conv_with_b = tf.nn.bias_add(conv,b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out

def maxpool_layer(conv,k=2):
    return tf.nn.max_pool(conv, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def model():
    x_reshaped = tf.reshape(x, shape=[-1,150,150,1])

    conv_out1 = conv_layer(x_reshaped, w1, b1)
    maxpool_out1 = maxpool_layer(conv_out1)
    norm1 = tf.nn.lrn(maxpool_out1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    conv_out2 = conv_layer(norm1, w2, b2)
    maxpool_out2 = maxpool_layer(conv_out2)
    norm2 = tf.nn.lrn(maxpool_out2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    maxpool_reshaped = tf.reshape(maxpool_out2, [-1,w3.get_shape().as_list()[0]])
    local = tf.add(tf.matmul(maxpool_reshaped, w3), b3)
    local_out = tf.nn.relu(local)

    out = tf.add(tf.matmul(local_out, w_out), b_out)
    return out

model_op = model()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model_op, y))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))