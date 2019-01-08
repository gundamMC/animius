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
            'filter_size_1': 3,
            'num_filter_1': 10,
            'pool_size_1': 2,
            'pool_type': 'max',
            'filter_size_2': 5,
            'num_filter_2': 15,
            'fully_connected_1': 128
        }

    def __init__(self):

        super().__init__()

        self.x = None
        self.y = None
        self.prediction = None
        self.train_op = None
        self.cost = None
        self.tb_merged = None

    def build_graph(self, model_config, data, graph=None):

        # make copies of the dictionaries since we will be editing it
        self.config = dict(model_config.config)
        self.config['class'] = 'SpeakerVerification'
        self.model_structure = dict(model_config.model_structure)
        self.hyperparameters = dict(model_config.hyperparameters)
        self.data = data

        def test_model_structure(key, lambda_value):
            if key in self.model_structure:
                return self.model_structure[key]
            elif self.data is None:
                raise ValueError('Data cannot be none')
            else:
                self.model_structure[key] = lambda_value()
                return lambda_value()

        if graph is None:
            graph = tf.Graph()

        with graph.as_default():

            input_window = test_model_structure('input_window', data.values['x'].shape[1])
            input_cepstral = test_model_structure('input_cepstral', data.values['x'].shape[2])

            # Tensorflow placeholders
            self.x = tf.placeholder(tf.float32,
                                    [None,
                                     input_window,
                                     input_cepstral],
                                    name='input_x'
                                    )
            self.y = tf.placeholder(tf.float32, [None, 1], name='train_y')

            # Network parameters
            weights = {
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
                'wd1': tf.Variable(tf.random_normal([round(self.model_structure['input_window'] / 2) *
                                                     round(self.model_structure['input_cepstral'] / 2) *
                                                     self.model_structure['num_filter_2'],
                                                     self.model_structure['fully_connected_1']]
                                                    )),
                # output, 128 input nodes, 1 output node
                'out': tf.Variable(tf.random_normal([128, 1]))
            }

            biases = {
                'bc1': tf.Variable(tf.random_normal([self.model_structure['num_filter_1']])),
                'bc2': tf.Variable(tf.random_normal([self.model_structure['num_filter_2']])),
                'bd1': tf.Variable(tf.random_normal([self.model_structure['fully_connected_1']])),
                'out': tf.Variable(tf.random_normal([1]))  # one output node
            }

            # define neural network
            def network():

                conv_x = tf.expand_dims(self.x, -1)

                conv1 = tf.nn.conv2d(conv_x, weights["wc1"], strides=[1, 1, 1, 1], padding='SAME')

                if self.model_structure['pool_type'] == 'max':
                    conv1_pooled = tf.nn.max_pool(conv1,
                                                  ksize=[1, self.model_structure['pool_size_1'],
                                                         self.model_structure['pool_size_1'],
                                                         1],
                                                  strides=[1, self.model_structure['pool_size_1'],
                                                           self.model_structure['pool_size_1'],
                                                           1],
                                                  padding='SAME')

                else:
                    conv1_pooled = tf.nn.avg_pool(conv1,
                                                  ksize=[1, self.model_structure['pool_size_1'],
                                                         self.model_structure['pool_size_1'],
                                                         1],
                                                  strides=[1, self.model_structure['pool_size_1'],
                                                           self.model_structure['pool_size_1'],
                                                           1],
                                                  padding='SAME')

                conv2 = tf.nn.conv2d(conv1_pooled, weights["wc2"], strides=[1, 1, 1, 1], padding='SAME')

                conv2 = tf.reshape(conv2, [tf.shape(self.x)[0], -1])  # maintain batch size
                fc1 = tf.add(tf.matmul(conv2, weights["wd1"]), biases["bd1"])
                fc1 = tf.nn.relu(fc1)

                out = tf.add(tf.matmul(fc1, weights['out']), biases['out'], name='output_predict')

                return out

            # Optimization
            self.prediction = network()
            self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=network(), labels=self.y),
                                       name='train_cost')
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.hyperparameters['learning_rate']).minimize(
                self.cost, name='train_op')

            # Tensorboard
            if self.config['tensorboard'] is not None:
                tf.summary.scalar('cost', self.cost)
                # tf.summary.scalar('accuracy', self.accuracy)
                self.tb_merged = tf.summary.merge_all(name='tensorboard_merged')

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

                    self.config['cost'] = cost_value

                    if self.config['hyperdash'] is not None:
                        self.hyperdash.metric("cost", cost_value)

                else:
                    self.sess.run([self.train_op], feed_dict={
                        self.x: batch_x,
                        self.y: batch_y
                    })

            if self.config['tensorboard'] is not None:
                summary = self.sess.run(self.tb_merged, feed_dict={
                    self.x: mini_batches_x[0],
                    self.y: mini_batches_y[0]
                })
                self.tensorboard_writer.add_summary(summary, self.config['epoch'])

            self.config['epoch'] += 1

    @classmethod
    def load(cls, directory, name='model', data=None):

        model = SpeakerVerificationModel()
        model.restore_config(directory, name)
        if data is None:
            model.data = data

        graph = tf.Graph()

        checkpoint = tf.train.get_checkpoint_state(directory)
        input_checkpoint = checkpoint.model_checkpoint_path

        with graph.as_default():
            model.saver = tf.train.import_meta_graph(input_checkpoint + '.meta')

        model.saver.restore(model.sess, input_checkpoint)

        # set up self attributes used by other methods
        model.x = model.sess.graph.get_tensor_by_name('input_x:0')
        model.y = model.sess.graph.get_tensor_by_name('train_y:0')
        model.train_op = model.sess.graph.get_operation_by_name('train_op')
        model.cost = model.sess.graph.get_tensor_by_name('train_cost:0')
        model.prediction = model.sess.graph.get_tensor_by_name('output_predict:0')

        model.init_tensorflow(graph)

        model.saved_directory = directory
        model.saved_name = name

        return model

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
