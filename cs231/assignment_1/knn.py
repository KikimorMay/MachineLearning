#from assignment_1.cs231n.features import color_histogram_hsv, hog_feature
import random
import numpy as np
from assignment_1.cs231n.data_utils import load_CIFAR10
import tensorflow as tf
import matplotlib.pyplot as plt

def get_CIFAR10_data(num_training = 49000, num_validation = 1000, num_test = 1000):
    cifar10_dir = 'cs231n/cifar-10-python'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    mask = list(range(num_training, num_training  + num_validation))
    X_val = np.array(X_train[mask].reshape(num_validation, 32*32*3), dtype = 'float32')
    y_val = np.array(y_train[mask])
    mask = list(range(num_training))
    X_train = np.array(X_train[mask].reshape(num_training, 32*32*3), dtype = 'float32')
    y_train = np.array(y_train[mask])
    mask = list(range(num_test))
    X_test = np.array(X_test[mask].reshape(num_test, 32*32*3), dtype = 'float32')
    y_test =  np.array(y_test[mask])

    return X_train, y_train, X_val, y_val, X_test, y_test

def prediction(X_test, W, b):
    y_test_pre = tf.nn.softmax(tf.matmul(X_test, W) + b)
    return y_test_pre

def evaluting(y_test_pre, y_test):
    correct_prediction = tf.equal(tf.argmax(y_test_pre, 1), tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


# X_train = tf.placeholder(tf.float32, [None, 32*32*3])
# y_train = tf.placeholder(tf.float32, [None, 10])
# W = tf.Variable(tf.zeros([32*32*3, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# multi_result = tf.matmul(X_train, W) + b
# pred_y = tf.nn.softmax(multi_result)
#
# loss = tf.reduce_mean(-tf.reduce_sum(y_train * tf.log(pred_y), reduction_indices=[1]))
# init = tf.global_variables_initializer()
#
# train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(loss)
#
# X_train_, y_train_, X_val_, y_val_, X_test_, y_test_ = get_CIFAR10_data()
#
#
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(5000):
#         batch_ys = np.zeros([1000, 10])   #每个batch 是100个样本
#         j = i%49
#         batch_xs = X_train_[j*1000:(j+1)*1000, :]
#         batch_ys_y = y_train_[j*1000:(j+1)*1000]
#         batch_ys[list(range(1000)), batch_ys_y] = 1
#         _, loss_v = sess.run([train_step, loss], feed_dict={X_train: batch_xs, y_train: batch_ys})
#         print('step %d loss=%f' % (i, loss_v))
# #        if i % 50 == 0:
# #            print(sess.run(multi_result, feed_dict={X_train: batch_xs, y_train: batch_ys}))
#         if i == 4999:
#             W_, b_ = sess.run([W, b], feed_dict={X_train: batch_xs, y_train: batch_ys})
#             y_test_pre = prediction(X_test_, W_, b_)
#             y_test__ = np.zeros([1000, 10])
#             y_test__[list(range(1000)), y_test_] = 1
#             accuracy = evaluting(y_test_pre, y_test__)
#             print(sess.run(accuracy))
#
