from assignment_1.cs231n.features import color_histogram_hsv, hog_feature
import random
import numpy as np
from assignment_1.cs231n.data_utils import load_CIFAR10
import tensorflow as tf
import matplotlib.pyplot as plt

def get_CIFAR10_data(num_training = 49000, num_validation = 1000, num_test = 1000):
    cifar10_dir = 'cs231n\\cifar-10-python'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    mask = list(range(num_training, num_training  + num_validation))
    X_val = np.array(X_train[mask].reshape(num_validation, 32*32*3), dtype = 'float32')
    y_val = np.array(y_train[mask], dtype = 'float32')
    mask = list(range(num_training))
    X_train = np.array(X_train[mask].reshape(num_training, 32*32*3), dtype = 'float32')
    y_train = np.array(y_train[mask])
    mask = list(range(num_test))
    X_test = np.array(X_test[mask].reshape(num_test, 32*32*3), dtype = 'float32')
    y_test =  np.array(y_test[mask], dtype = 'float32')

    return X_train, y_train, X_val, y_val, X_test, y_test




X_train = tf.placeholder(tf.float32, [None, 32*32*3])
y_train = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([32*32*3, 10]))
b = tf.Variable(tf.zeros([10]))

multi_result = tf.matmul(X_train, W) + b
pred_y = tf.nn.softmax(multi_result)

loss = tf.reduce_mean(-tf.reduce_sum(y_train * tf.log(pred_y), reduction_indices=[1]))
init = tf.global_variables_initializer()

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
X_train_, y_train_, X_val_, y_val_, X_test_, y_test_ = get_CIFAR10_data()


with tf.Session() as sess:
    sess.run(init)
    for i in range(5000):
        batch_ys = np.zeros([100, 10])   #每个batch 是100个样本
        j = i%490
        batch_xs = X_train_[j*100:(j+1)*100, :]
        batch_ys_y = y_train_[j*100:(j+1)*100]
        batch_ys[list(range(100)), batch_ys_y] = 1
        sess.run(train_step, feed_dict={X_train: batch_xs, y_train: batch_ys})

        if i % 50 == 0:
            print(sess.run(multi_result, feed_dict={X_train: batch_xs, y_train: batch_ys}))



