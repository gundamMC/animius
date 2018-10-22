import tensorflow as tf
import numpy as np
from ProjectWaifu.Model import Model
from ProjectWaifu.Utils import get_mini_batches, shuffle
import os


class SpeakerVerificationModel(Model):

    # default values
    @staticmethod
    def DEFAULT_HYPERPARAMETERS():
        return {
            'learning_rate': 0.005,
            'batch_size': 2048,
            'optimizer': 'adam'
        }

    @staticmethod
    def DEFAULT_MODEL_STRUCTURE():
        return {
            'input_window': 10,
            'input_cepstral': 39,
            'filter_size_1': 3,
            'num_filter_1': 10,
            'pool_size_1': 2,
            'pool_type': 'max',
            'filter_size_2': 5,
            'num_filter_2': 15,
            'fully_connected_1': 128
        }

    def __init__(self, model_config, data, restore_path=None):

        super().__init__(model_config, data, restore_path=restore_path)

        # Tensorflow placeholders
        self.x = tf.placeholder(tf.float32, [None,
                                             self.model_structure['input_window'],
                                             self.model_structure['input_cepstral'],
                                             1])
        self.y = tf.placeholder(tf.float32, [None])

        # Network parameters
        self.weights = {
            # 3x3 conv filter, 1 input layers, 10 output layers
            'wc1': tf.Variable(tf.random_normal([self.model_structure['filter_size_1'],
                                                 self.model_structure['filter_size_1'],
                                                 1,
                                                 self.model_structure['num_filter_1']]
                                                )),
            # 5x5 conv filter, 10 input layers, 15 output layers
            'wc2': tf.Variable(tf.random_normal([self.model_structure['filter_size_2'],
                                                 self.model_structure['filter_size_2'],
                                                 self.model_structure['num_filter_1'],
                                                 self.model_structure['num_filter_2']]
                                                )),
            # fully connected 1, 15 input layers, 128 outpute nodes
            'wd1': tf.Variable(tf.random_normal([int(self.model_structure['input_window']/2) *
                                                 int(self.model_structure['input_cepstral']/2) *
                                                 self.model_structure['num_filter_2'],
                                                 self.model_structure['fully_connected_1']]
                                                )),
            # output, 128 input nodes, 1 output node
            'out': tf.Variable(tf.random_normal([128, 1]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([self.model_structure['num_filter_1']])),
            'bc2': tf.Variable(tf.random_normal([self.model_structure['num_filter_2']])),
            'bd1': tf.Variable(tf.random_normal([self.model_structure['fully_connected_1']])),
            'out': tf.Variable(tf.random_normal([1]))  # one output node
        }

        # Optimization
        self.prediction = tf.nn.sigmoid(self.network(self.x))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.network(self.x),
                                                                              labels=self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.hyperparameters['learning_rate']).minimize(self.cost)

        # Tensorboard
        if self.config['tensorboard'] is not None:
            tf.summary.scalar('cost', self.cost)
            # tf.summary.scalar('accuracy', self.accuracy)
            self.merged = tf.summary.merge_all()

        self.init_tensorflow()

        self.init_hyerdash(self.config['hyperdash'])

        # restore model data values
        self.init_restore(restore_path)

    def network(self, X):

        conv1 = tf.nn.conv2d(X, self.weights["wc1"], strides=[1, 1, 1, 1], padding='SAME')

        if self.model_structure['pool_type'] == 'max':
            conv1 = tf.nn.max_pool(conv1,
                                   ksize=[1, self.model_structure['pool_size_1'], self.model_structure['pool_size_1'],
                                          1],
                                   strides=[1, self.model_structure['pool_size_1'], self.model_structure['pool_size_1'],
                                            1],
                                   padding='SAME')

        elif self.model_structure['pool_type'] == 'avg':
            conv1 = tf.nn.avg_pool(conv1,
                                   ksize=[1, self.model_structure['pool_size_1'], self.model_structure['pool_size_1'],
                                          1],
                                   strides=[1, self.model_structure['pool_size_1'], self.model_structure['pool_size_1'],
                                            1],
                                   padding='SAME')

        conv2 = tf.nn.conv2d(conv1, self.weights["wc2"], strides=[1, 1, 1, 1], padding='SAME')

        fc1 = tf.reshape(conv2, [-1])
        fc1 = tf.add(tf.matmul(fc1, self.weights["wd1"]), self.biases["bd3"])
        fc1 = tf.nn.relu(fc1)

        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

        # softmax is applied during tf.nn.softmax_cross_entropy_with_logits
        return out

    def train(self, epochs=800):

        for epoch in range(epochs):

            mini_batches_x, mini_batches_y = get_mini_batches(
                shuffle([
                    self.data['x'],
                    self.data['x_length'],
                    self.data['y_intent'],
                    self.data['y_ner']]
                ), self.hyperparameters['batch_size'])

            for batch in range(len(mini_batches_x)):
                batch_x = mini_batches_x[batch]
                batch_y = mini_batches_y[batch]

                if (self.config['epoch'] % self.config['display_step'] == 0 or self.config['display_step'] == 0) \
                        and (batch % 100 == 0 or batch == 0):
                    _, cost_value = self.sess.run([self.train_op, self.cost], feed_dict={
                        self.x: batch_x,
                        self.y: batch_y
                    })

                    print("epoch:", self.config['epoch'], "- (", batch, "/", len(mini_batches_x), ") -", cost_value)

                    if self.config['hyperdash']:
                        self.hyperdash.metric("cost", cost_value)

                else:
                    self.sess.run([self.train_op], feed_dict={
                        self.x: batch_x,
                        self.y: batch_y
                    })

            if self.config['tensorboard'] is not None:
                summary = self.sess.run(self.merged, feed_dict={
                    self.x: mini_batches_x[0],
                    self.y: mini_batches_y[0]
                })
                self.tensorboard_writer.add_summary(summary, self.config['epoch'])

    def predict(self, input_data, save_path=None):
        result = self.sess.run(self.prediction,
                               feed_dict={
                                   self.x: input_data.values['x'],
                               })

        if save_path is not None:
            with open(save_path, "w") as file:
                for i in range(len(result)):
                    file.write(str(result[i]) + '\n')

        return result

    #
    # @staticmethod
    # def getData(TruePaths, FalsePaths=None):
    #     x0 = np.empty(shape=[0, 10, 39])
    #     x1 = np.empty(shape=[0, 10, 39])
    #
    #     for path in TruePaths:
    #         x0 = np.append(x0, getMFCC(path, False), axis=0)
    #
    #     if FalsePaths is None:
    #         return x0[..., np.newaxis]
    #
    #     for path in FalsePaths:
    #         x1 = np.append(x1, getMFCC(path, False), axis=0)
    #
    #     y0 = np.tile([1,0], (x0.shape[0],1))
    #     y1 = np.tile([0,1], (x1.shape[0],1))
    #
    #     datax = np.append(x0, x1, axis=0)
    #     datay = np.append(y0, y1, axis=0)
    #
    #     datax = datax[..., np.newaxis]
    #
    #     return datax, datay

    # def setTrainingData(self, dataPaths):
    #     # dataPaths[0] = True Path
    #     # dataPaths[1] = False Path
    #     TrainTruePaths = [line.strip() for line in open(dataPaths[0], encoding='utf-8')]
    #     TrainFalsePaths = [line.strip() for line in open(dataPaths[1], encoding='utf-8')]
    #     self.train_x, self.train_y = self.getData(TrainTruePaths, TrainFalsePaths)
    #     self.data_set = True
