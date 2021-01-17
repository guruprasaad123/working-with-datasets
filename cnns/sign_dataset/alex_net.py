
import tensorflow as tf

def get_weights_biases(n_classes):
    # Store layers weight & bias
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
        'wd2': tf.Variable(tf.random_normal([1024, 1024])),
        'out': tf.Variable(tf.random_normal([1024, 10]))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([128])),
        'bc3': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'bd2': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # returns weights & biases dict
    return (weights,biases)

