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

def change_y_to_matrix(y_vector):
    num = y_vector.shape[0]
    y_matrix = np.zeros([num, 10])
    y_matrix[list(range(num)), y_vector] = 1
    return y_matrix

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
y_train_ = change_y_to_matrix(y_train)
y_test_ = change_y_to_matrix(y_test)

X_batch = tf.placeholder(tf.float32, [None, 3072])
Y_batch = tf.placeholder(tf.float32, [None, 10])

output1 = layer(X_batch, 3072, 200, tf.nn.tanh)
output = layer(output1, 200, 10)


y_softmax_pre = tf.nn.softmax(output)
loss = tf.reduce_mean(tf.reduce_sum(-Y_batch * tf.log(y_softmax_pre), reduction_indices=[1]))
accuracy = predition(y_softmax_pre, Y_batch)
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batch_size = 100
    epoch = 100
    loss_train = []; acc_train = []; loss_test = []; acc_test = []
    for i in range(epoch):

        for j in range(X_train.shape[0]//batch_size):
            X_batch_ = X_train[j*batch_size: (j+1)*batch_size, :]
            Y_batch_ = y_train_[j*batch_size: (j+1)*batch_size, :]
            sess.run(train_step, feed_dict={X_batch: X_batch_, Y_batch:Y_batch_})
        loss_, acc = sess.run([loss, accuracy], feed_dict={X_batch: X_train, Y_batch: y_train_})
        loss_2, acc_2 = sess.run([loss, accuracy], feed_dict={X_batch: X_test, Y_batch: y_test_})

        loss_train.append(loss_); acc_train.append(acc)
        loss_test.append(loss_2); acc_test.append(acc_2)

        print('the training set:%d: loss:%f, acc = %f' % (i, loss_, acc))
        print('the test set:%d: loss:%f, acc:%f\n' %(i, loss_2, acc_2))
np.save('loss_train.npy', loss_train)
np.save('acc_train.npy', acc_train)
np.save('loss_test.npy', loss_test)
np.save('acc_test.npy', acc_test)

x = list(range(100))

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x,loss_train,'b-',label='loss_train') # in 'bo-', b is blue, o is O marker, - is solid line and so on
ax1.plot(x,loss_test,'y-',label='loss_test')
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x,acc_train,'g-',label='acc_train')
ax2.plot(x,acc_test,'c-',label='acc_test')

plt.show()