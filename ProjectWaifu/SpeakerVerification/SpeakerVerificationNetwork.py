import tensorflow as tf
import numpy as np
from ProjectWaifu.SpeakerVerification.MFCC import getMFCC
from ProjectWaifu.Network import Network
from ProjectWaifu import Utils


class SpeakerVerificationNetwork(Network):

    def __init__(self, learning_rate=0.01, batch_size=1024):
        # hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network hyperparameters
        # Input: 10x39x1
        self.filter_size_1 = 3
        self.num_filter_1 = 10
        self.padding_1 = "SAME"
        # Filter 1: 10x39x10
        self.max_pool_size_1 = 2
        self.padding_pool = 'SAME'
        # Pooling 1: 5x20x10
        self.filter_size_2 = 5
        self.num_filter_2 = 15
        self.padding_2 = "SAME"
        # Filter 2: 5x20x15
        self.fully_connected_1 = 128
        self.softmax_output = 2

        # Tensorflow placeholders
        self.x = tf.placeholder(tf.float32, [None, 10, 39, 1])
        self.y = tf.placeholder(tf.float32, [None, 2])

        # Network parameters
        self.weights = {
            # 3x3 conv filter, 1 input layers, 10 output layers
            'wc1': tf.Variable(tf.random_normal([self.filter_size_1, self.filter_size_1, 1, self.num_filter_1])),
            # 5x5 conv filter, 10 input layers, 15 output layers
            'wc2': tf.Variable(tf.random_normal([self.filter_size_2, self.filter_size_2, self.num_filter_1, self.num_filter_2])),
            # fully connected 1, 15 input layers, 128 outpute nodes
            'wd1': tf.Variable(tf.random_normal([5 * 20 * 15, self.fully_connected_1])),
            # output, 128 input nodes, 2 output nodes
            'out': tf.Variable(tf.random_normal([128, self.softmax_output]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([self.num_filter_1])),
            'bc2': tf.Variable(tf.random_normal([self.num_filter_2])),
            'bd3': tf.Variable(tf.random_normal([self.fully_connected_1])),
            'out': tf.Variable(tf.random_normal([self.softmax_output]))
        }

        # Optimization
        self.prediction = tf.nn.softmax(self.network(self.x))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.network(self.x),
                                                                              labels=self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Tensorflow initialization
        self.sess = tf.Session(config=self.config)

        self.sess.run(tf.global_variables_initializer())

        self.data_set = False
        self.train_x = None
        self.train_y = None

    @staticmethod
    def getData(TruePaths, FalsePaths=None):
        x0 = np.empty(shape=[0, 10, 39])
        x1 = np.empty(shape=[0, 10, 39])

        for path in TruePaths:
            x0 = np.append(x0, getMFCC(path, False), axis=0)

        if FalsePaths is None:
            return x0[..., np.newaxis]

        for path in FalsePaths:
            x1 = np.append(x1, getMFCC(path, False), axis=0)

        y0 = np.tile([1,0], (x0.shape[0],1))
        y1 = np.tile([0,1], (x1.shape[0],1))

        datax = np.append(x0, x1, axis=0)
        datay = np.append(y0, y1, axis=0)

        datax = datax[..., np.newaxis]

        return datax, datay

    def setTrainingData(self, TruePath, FalsePath):
        TrainTruePaths = [line.strip() for line in open(TruePath, encoding='utf-8')]
        TrainFalsePaths = [line.strip() for line in open(FalsePath, encoding='utf-8')]
        self.train_x, self.train_y = self.getData(TrainTruePaths, TrainFalsePaths)

    def network(self, X):

        conv1 = tf.nn.conv2d(X, self.weights["wc1"], strides=[1, 1, 1, 1], padding=self.padding_1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, self.max_pool_size_1, self.max_pool_size_1, 1], strides=[1, 2, 2, 1],
                               padding=self.padding_pool)

        conv2 = tf.nn.conv2d(conv1, self.weights["wc2"], strides=[1, 1, 1, 1], padding=self.padding_2)

        fc1 = tf.reshape(conv2, [-1, 5 * 20 * 15])
        fc1 = tf.add(tf.matmul(fc1, self.weights["wd1"]), self.biases["bd3"])
        fc1 = tf.nn.relu(fc1)

        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

        # softmax is applied during tf.nn.softmax_cross_entropy_with_logits
        return out

    def train(self, epochs=50, display_step=10):
        if not self.data_set:
            print("Error: Training data not set")
            return

        # Start training
        print("starting training")
        for epoch in range(epochs + 1):  # since range is exclusive

            mini_batches_X, mini_batches_Y =\
                Utils.random_mini_batches([self.train_x, self.train_y], int(len(self.train_x) / self.batch_size))

            for i in range(0, len(mini_batches_X)):
                batch_x = mini_batches_X[i]
                batch_y = mini_batches_Y[i]

                self.sess.run(self.train_op, feed_dict={self.x: batch_x,
                                                        self.y: batch_y})

                if epoch % display_step == 0:
                    cost_value = self.sess.run([self.cost],
                                               feed_dict={self.x: batch_x,
                                                          self.y: batch_y})
                    print('epoch', epoch, '(', i, '/', len(mini_batches_X), ') - cost', cost_value)

    def predict(self, path):
        X = self.getData([path])
        result = self.sess.run(self.prediction, feed_dict={self.x: X})
        return np.sum(result, axis=0) / result.shape[0]
