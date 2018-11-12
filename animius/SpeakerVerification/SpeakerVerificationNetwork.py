import tensorflow as tf
import animius as am


class SpeakerVerificationModel(am.Model):

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

        graph = tf.Graph()
        with graph.as_default():
            # Tensorflow placeholders
            self.x = tf.placeholder(tf.float32, [None,
                                                 self.model_structure['input_window'],
                                                 self.model_structure['input_cepstral']])
            self.y = tf.placeholder(tf.float32, [None, 1])

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
                'wd1': tf.Variable(tf.random_normal([round(self.model_structure['input_window']/2) *
                                                     round(self.model_structure['input_cepstral']/2) *
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
            self.prediction = self.network()
            self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.network(), labels=self.y))
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

    def network(self):

        conv_x = tf.expand_dims(self.x, -1)

        conv1 = tf.nn.conv2d(conv_x, self.weights["wc1"], strides=[1, 1, 1, 1], padding='SAME')

        if self.model_structure['pool_type'] == 'max':
            conv1_pooled = tf.nn.max_pool(conv1,
                                   ksize=[1, self.model_structure['pool_size_1'], self.model_structure['pool_size_1'],
                                          1],
                                   strides=[1, self.model_structure['pool_size_1'], self.model_structure['pool_size_1'],
                                            1],
                                   padding='SAME')

        else:
            conv1_pooled = tf.nn.avg_pool(conv1,
                                   ksize=[1, self.model_structure['pool_size_1'], self.model_structure['pool_size_1'],
                                          1],
                                   strides=[1, self.model_structure['pool_size_1'], self.model_structure['pool_size_1'],
                                            1],
                                   padding='SAME')

        conv2 = tf.nn.conv2d(conv1_pooled, self.weights["wc2"], strides=[1, 1, 1, 1], padding='SAME')

        conv2 = tf.reshape(conv2, [tf.shape(self.x)[0], -1])  # maintain batch size
        fc1 = tf.add(tf.matmul(conv2, self.weights["wd1"]), self.biases["bd1"])
        fc1 = tf.nn.relu(fc1)

        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

        return out

    def train(self, epochs=800):

        for epoch in range(epochs):

            mini_batches_x, mini_batches_y = am.Utils.get_mini_batches(
                am.Utils.shuffle([
                    self.data['x'],
                    self.data['y']]
                ), self.hyperparameters['batch_size'])

            for batch in range(len(mini_batches_x)):
                batch_x = mini_batches_x[batch]
                batch_y = mini_batches_y[batch]

                if (self.config['display_step'] == 0 or
                    self.config['epoch'] % self.config['display_step'] == 0 or
                    epoch == epochs) and \
                        (batch % 100 == 0 or batch == 0):
                    _, cost_value = self.sess.run([self.train_op, self.cost], feed_dict={
                        self.x: batch_x,
                        self.y: batch_y
                    })

                    print("epoch:", self.config['epoch'], "- (", batch, "/", len(mini_batches_x), ") -", cost_value)

                    if self.config['hyperdash'] is not None:
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

            self.config['epoch'] += 1

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


# modelConfig = SpeakerVerificationModel.DEFAULT_MODEL_CONFIG()
# modelConfig.model_structure['input_window'] = 50
# modelConfig.config['display_step'] = 20
#
# data = ModelClasses.SpeakerVerificationData(modelConfig)
#
# data.parse_data_file('D:\Project Waifu\Project-Waifu\animius\\audio\\True.txt', output=True)
# data.parse_data_file('D:\Project Waifu\Project-Waifu\animius\\audio\\False.txt', output=False)
# model = SpeakerVerificationModel(modelConfig, data)
#
# test = ModelClasses.SpeakerVerificationData(modelConfig)
# test.parse_input_file('D:\Project Waifu\Project-Waifu\animius\\audio\Hyouka - 01\\0020.wav')
#
# model.train(150)
# model.save()
#
# import numpy as np
#
# print(np.mean(model.predict(test)))
