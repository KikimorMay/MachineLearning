from assignment_1.cs231n.features import color_histogram_hsv, hog_feature
import random
import numpy as np
from knn import get_CIFAR10_data
import tensorflow as tf
import matplotlib.pyplot as plt

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
# X_train.shape (49000, 3072
# y_train.shape (49000,
# X_val, X_test.shape (1000, 3072
# y_val, y_test,shape (1000,


def layer(inputs, insize, outsize, activation_function = None):
    W = tf.Variable(tf.random_normal([insize, outsize]))
    b = tf.Variable(tf.random_normal([outsize]))
    Wx_plus_b = tf.matmul(inputs, W) + b

    if activation_function :
        output = activation_function(Wx_plus_b)
    else:
        output = Wx_plus_b
    return output

def predition(y_softmax_pre , y_test):
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(y_softmax_pre, 1), tf.argmax(y_test, 1)), tf.float32)
    )
    return accuracy


X_batch = tf.placeholder(tf.float32, [None, 3072])
Y_batch = tf.placeholder(tf.float32, [None, 10])

output1 = layer(X_batch, 3072, 100, tf.nn.tanh)
output = layer(output1, 100, 10)

y_softmax_pre = tf.nn.softmax(output)
loss = tf.reduce_mean(tf.reduce_sum(-Y_batch * tf.log(y_softmax_pre), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batch_size = 1000
    step_size = 1000
    for i in range(step_size):
        j = i%(49000//batch_size)
        X_batch_ = X_train[j*batch_size: (j+1)*batch_size, :]
        Y_batch_y = y_train[j*batch_size: (j+1)*batch_size]
        Y_batch_ = np.zeros([batch_size, 10])
        Y_batch_[list(range(batch_size)), Y_batch_y] = 1
        if i%50 == 0:
            accuracy = predition(y_softmax_pre , Y_batch_)
            loss_, acc = sess.run([loss, accuracy], feed_dict={X_batch: X_batch_, Y_batch:Y_batch_})
            print('%d: loss:%f, acc = %f' %(i, loss_, acc))
        sess.run(train_step, feed_dict={X_batch: X_batch_, Y_batch:Y_batch_})


