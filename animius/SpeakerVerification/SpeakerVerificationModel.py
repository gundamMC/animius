import tensorflow as tf

import animius as am


class SpeakerVerificationModel(am.Model):

    # default values
    @staticmethod
    def DEFAULT_HYPERPARAMETERS():
        return {
            'learning_rate': 0.001,
            'batch_size': 512,
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
            'fully_connected_1': 128,
            'input_window': 20,
            'input_cepstral': 39
        }

    def __init__(self):

        super().__init__()

        self.x = None
        self.y = None
        self.prediction = None
        self.train_op = None
        self.cost = None
        self.tb_merged = None

        self.data_count = None
        self.iterator = None
        self.predict_dataset = None
        self.predict_iterator = None

    def init_dataset(self, data=None):

        super().init_dataset(data)

        self.data_count = tf.placeholder(tf.int64, shape=(), name='ds_data_count')

        index_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(tf.range(self.data_count), -1))

        ds = index_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.data.steps_per_epoch))

        def _py_func(x):
            return tf.py_func(self.data.parse, [x, False], [tf.float32, tf.float32])

        ds = ds.map(_py_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.apply(tf.data.experimental.unbatch())  # testing needed

        ds = ds.batch(batch_size=self.hyperparameters['batch_size'])

        ds = ds.apply(tf.data.experimental.prefetch_to_device(self.config['device'],
                                                              buffer_size=tf.data.experimental.AUTOTUNE))

        self.dataset = ds

        return ds

    def init_predict_dataset(self):
        index_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(tf.range(self.data_count), -1))

        def _py_func(x):
            return tf.py_func(self.data.parse, [x, True], tf.float32)

        ds = index_ds.map(_py_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.apply(tf.data.experimental.unbatch())

        ds = ds.batch(batch_size=self.hyperparameters['batch_size'])

        ds = ds.apply(tf.data.experimental.prefetch_to_device(self.config['device'],
                                                              buffer_size=tf.data.experimental.AUTOTUNE))

        self.predict_dataset = ds

    def build_graph(self, model_config, data, graph=None):

        # make copies of the dictionaries since we will be editing it
        self.config = dict(model_config.config)
        self.config['class'] = 'SpeakerVerification'
        self.model_structure = dict(model_config.model_structure)
        self.hyperparameters = dict(model_config.hyperparameters)
        self.data = data

        if graph is None:
            graph = tf.Graph()

        with graph.as_default():

            if 'GPU' in self.config['device'] and not tf.test.is_gpu_available():
                self.config['device'] = '/cpu:0'
                # override to CPU since no GPU is available

            with graph.device('/cpu:0'):
                if self.dataset is None:
                    self.init_dataset(data)
                self.iterator = self.dataset.make_initializable_iterator()

                if self.predict_dataset is None:
                    self.init_predict_dataset()
                self.predict_iterator = self.predict_dataset.make_initializable_iterator()

            with graph.device(self.config['device']):

                self.x, self.y = self.iterator.get_next()

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
                def network(x):

                    conv_x = tf.expand_dims(x, -1)

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

                    conv2 = tf.reshape(conv2, [tf.shape(x)[0], -1])  # maintain batch size
                    fc1 = tf.add(tf.matmul(conv2, weights["wd1"]), biases["bd1"])
                    fc1 = tf.nn.relu(fc1)

                    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'], name='output_predict')

                    return out

                # Optimization
                self.prediction = network(self.predict_iterator.get_next())
                self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=network(self.x),
                                                                                   labels=self.y),
                                           name='train_cost')
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.hyperparameters['learning_rate']).minimize(
                    self.cost, name='train_op')

                # Tensorboard
                if self.config['tensorboard'] is not None:
                    tf.summary.scalar('cost', self.cost)
                    # tf.summary.scalar('accuracy', self.accuracy)
                    self.tb_merged = tf.summary.merge_all(name='tensorboard_merged')

        self.graph = graph
        return graph

    def train(self, epochs=800, cancellation_token=None):

        print('starting training')

        with self.graph.device('/cpu:0'):
            self.sess.run(self.iterator.initializer, feed_dict={self.data_count: len(self.data['train_y'])})

        print('initialized iterator')

        epoch = 0

        while epoch < epochs:

            print('training epoch', epoch, '|', self.data.steps_per_epoch)

            if cancellation_token is not None and cancellation_token.is_cancalled:
                return  # early stopping

            batch_num = 0

            try:

                while batch_num < self.data.steps_per_epoch:

                    if (self.config['display_step'] == 0 or
                        self.config['epoch'] % self.config['display_step'] == 0 or
                        epoch == epochs) and \
                            (batch_num % 100 == 0 or batch_num == 0):
                        _, cost_value = self.sess.run([self.train_op, self.cost])

                        print("epoch:", self.config['epoch'], "- (", batch_num, ") -", cost_value)

                        self.config['cost'] = cost_value.item()

                        if self.config['hyperdash'] is not None:
                            self.hyperdash.metric("cost", cost_value)

                    else:
                        self.sess.run([self.train_op])

                    batch_num += 1

            except tf.errors.OutOfRangeError:
                # this should never happen
                print(batch_num)

            epoch += 1

            if self.config['tensorboard'] is not None:
                summary = self.sess.run(self.tb_merged)
                self.tensorboard_writer.add_summary(summary, self.config['epoch'])

            self.config['epoch'] += 1

    @classmethod
    def load(cls, directory, name='model', data=None):

        model = SpeakerVerificationModel()
        model.restore_config(directory, name)
        if data is not None:
            model.data = data
        else:
            model.data = am.SpeakerVerificationData()

        model.build_graph(model.model_config(), model.data)

        # model.sess = tf.Session(config=config, graph=graph)
        model.init_tensorflow(init_param=False, init_sess=True)

        checkpoint = tf.train.get_checkpoint_state(directory)
        input_checkpoint = checkpoint.model_checkpoint_path

        with model.graph.as_default():
            # model.init_dataset()
            # model.iterator = model.dataset.make_initializable_iterator()
            # model.iter_init_op = model.iterator.make_initializer(model.dataset, name='iterator_init')
            #
            # model.sess.run(model.iterator.initializer, feed_dict={model.data_count: len(model.data['train_y'])})

            # model.saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
            #                                          input_map={'IteratorGetNext': tf.convert_to_tensor(model.iterator.get_next())})
            model.saver.restore(model.sess, input_checkpoint)

        # set up self attributes used by other methods
        # model.data_count = model.sess.graph.get_tensor_by_name('ds_data_count:0')
        # model.train_op = model.sess.graph.get_operation_by_name('train_op')
        # model.cost = model.sess.graph.get_tensor_by_name('train_cost:0')
        # model.prediction = model.sess.graph.get_tensor_by_name('output_predict:0')

        # model.init_tensorflow(graph, init_param=False, init_sess=False)

        model.saved_directory = directory
        model.saved_name = name

        return model

    def predict(self, input_data=None, save_path=None, raw=False):

        if input_data is None:
            input_data = self.data
        elif isinstance(input_data, am.SpeakerVerificationData):
            self.data = input_data  # is a new speaker verification data, override the current one
        else:
            self.data.set_wav_file(input_data, is_speaker=None)  # None = input
            input_data = self.data

        with self.graph.device('/cpu:0'):
            self.sess.run(self.predict_iterator.initializer, feed_dict={self.data_count: len(input_data['input'])})

        outputs = []
        batch_num = 0
        try:
            while batch_num < self.data.predict_steps:
                outputs.append(self.sess.run(self.prediction))
                batch_num += 1
        except tf.errors.OutOfRangeError:
            print(batch_num)

        import numpy as np
        outputs = np.concatenate(outputs)

        results = []
        end_index = 0

        print(outputs.shape)

        if raw:
            for index in range(len(self.data.values['input'])):
                windows = outputs[end_index:end_index + self.data.predict_step_nums[index]]
                end_index += self.data.predict_step_nums[index]
                results.append(windows.mean())
        else:
            for index in range(len(self.data.values['input'])):
                windows = outputs[end_index:end_index + self.data.predict_step_nums[index]]
                end_index += self.data.predict_step_nums[index]
                results.append(windows.mean() > 0.5)

        if save_path is not None:
            with open(save_path, "w") as file:
                for i in results:
                    file.write(str(i) + '\n')

        return results
