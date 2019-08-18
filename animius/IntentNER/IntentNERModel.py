import tensorflow as tf

import animius as am


class IntentNERModel(am.Model):

    # default values
    @staticmethod
    def DEFAULT_HYPERPARAMETERS():
        return {
            'learning_rate': 0.003,
            'batch_size': 1024,
            'optimizer': 'adam'
        }

    @staticmethod
    def DEFAULT_MODEL_STRUCTURE():
        return {
            'max_sequence': 20,
            'n_hidden': 128,
            'gradient_clip': 5.0,
            'node': 'gru',
            'n_intent_output': 15,
            'n_ner_output': 8
        }

    def __init__(self):

        super().__init__()

        self.x = None
        self.x_length = None
        self.y_intent = None
        self.y_ner = None
        self.prediction = None
        self.train_op = None
        self.cost = None
        self.tb_merged = None
        self.word_embedding = None

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
            x, x_length, y_intent, y_ner = tf.py_func(self.data.parse, [x, False], [tf.int32, tf.int32, tf.int32, tf.int32])
            y_intent = tf.one_hot(y_intent, self.model_structure['n_intent_output'])
            y_ner = tf.one_hot(y_ner, self.model_structure['n_ner_output'])
            return x, x_length, y_intent, y_ner

        ds = ds.apply(tf.data.experimental.map_and_batch(_py_func,
                                                         self.hyperparameters['batch_size'],
                                                         num_parallel_calls=tf.data.experimental.AUTOTUNE))

        ds = ds.apply(tf.data.experimental.prefetch_to_device(self.config['device'],  # preload to training device
                                                              buffer_size=tf.data.experimental.AUTOTUNE))

        self.dataset = ds

        return ds

    def init_predict_dataset(self):
        index_ds = tf.data.Dataset.from_tensor_slices(tf.expand_dims(tf.range(self.data_count), -1))

        def _py_func(x):
            return tf.py_func(self.data.parse, [x, True], [tf.int32, tf.int32])

        ds = index_ds.apply(tf.data.experimental.map_and_batch(_py_func,
                                                               self.hyperparameters['batch_size'],
                                                               num_parallel_calls=tf.data.experimental.AUTOTUNE))

        ds = ds.apply(tf.data.experimental.prefetch_to_device(self.config['device'],
                                                              buffer_size=tf.data.experimental.AUTOTUNE))

        self.predict_dataset = ds

        return ds

    def build_graph(self, model_config, data, graph=None, embedding_tensor=None):

        # make copies of the dictionaries since we will be editing it
        self.config = dict(model_config.config)
        self.config['class'] = 'IntentNER'
        self.model_structure = dict(model_config.model_structure)
        self.hyperparameters = dict(model_config.hyperparameters)
        self.data = data

        def test_model_structure(key, lambda_value):
            if key in self.model_structure:
                return self.model_structure[key]
            else:
                if self.data is None or 'embedding' not in self.data.values:
                    raise ValueError('When creating a new model, data must contain a word embedding')
                self.model_structure[key] = lambda_value()
                return lambda_value()

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

                if embedding_tensor is None:
                    n_vector = test_model_structure('n_vector', lambda: len(self.data["embedding"].embedding[0]))
                    word_count = test_model_structure('word_count', lambda: len(self.data["embedding"].words))
                    self.word_embedding = tf.Variable(tf.constant(0.0, shape=(word_count, n_vector)),
                                                      trainable=False, name='word_embedding')
                else:
                    self.word_embedding = embedding_tensor

                # Tensorflow placeholders
                self.x, self.x_length, self.y_intent, self.y_ner = self.iterator.get_next()
                self.x.set_shape([None, self.model_structure['max_sequence']])

                # Network parameters
                weights = {  # LSTM weights are created automatically by tensorflow
                    "out_intent": tf.Variable(
                        tf.random_normal([self.model_structure['n_hidden'], self.model_structure['n_intent_output']])),
                    "out_ner": tf.Variable(tf.random_normal(
                        [self.model_structure['n_hidden'] + self.model_structure['n_intent_output'],
                         self.model_structure['n_ner_output']]))
                }

                biases = {
                    "out_intent": tf.Variable(tf.random_normal([self.model_structure['n_intent_output']])),
                    "out_ner": tf.Variable(tf.random_normal([self.model_structure['n_ner_output']]))
                }

                cell_fw = tf.contrib.rnn.GRUCell(self.model_structure['n_hidden'])
                cell_bw = tf.contrib.rnn.GRUCell(self.model_structure['n_hidden'])

                # Setup model network

                def network(x, x_length):

                    embedded_x = tf.nn.embedding_lookup(self.word_embedding, x)
                    # manually give shape since the py_func in tf.data pipeline pretty much fucked up the static shape
                    embedded_x.set_shape([None, self.model_structure['max_sequence'], n_vector])

                    batch_size = tf.shape(x)[0]

                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                 cell_bw,
                                                                 inputs=embedded_x,
                                                                 dtype=tf.float32,
                                                                 sequence_length=x_length,
                                                                 swap_memory=True)
                    # see https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
                    # swap_memory is optional.
                    outputs_fw, output_bw = outputs  # outputs is a tuple (output_fw, output_bw)

                    # get last time steps
                    indexes = tf.reshape(tf.range(0, batch_size), [batch_size, 1])
                    last_time_steps = tf.reshape(tf.add(x_length, -1), [batch_size, 1])
                    last_time_step_indexes = tf.concat([indexes, last_time_steps], axis=1)

                    # apply linear
                    outputs_intent = tf.add(
                        tf.matmul(
                            tf.gather_nd(outputs_fw, last_time_step_indexes),
                            weights["out_intent"]
                        ),
                        biases["out_intent"]
                    )

                    entities = tf.concat(
                        [output_bw,
                         tf.tile(tf.expand_dims(outputs_intent, 1), [1, self.model_structure['max_sequence'], 1])], -1
                    )
                    outputs_entities = tf.add(
                        tf.einsum('ijk,kl->ijl', entities, weights["out_ner"]),
                        biases["out_ner"]
                    )

                    return outputs_intent, outputs_entities  # linear/no activation as there will be a softmax layer

                # Optimization
                logits_intent, logits_ner = network(self.x, self.x_length)
                self.cost = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_intent, labels=self.y_intent)
                ) + tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_ner, labels=self.y_ner),
                    name='train_cost'
                )

                # gradient clip rnn
                optimizer = tf.train.AdamOptimizer(self.hyperparameters['learning_rate'])
                gradients, variables = zip(*optimizer.compute_gradients(self.cost))
                gradients, _ = tf.clip_by_global_norm(gradients, self.model_structure['gradient_clip'])
                self.train_op = optimizer.apply_gradients(zip(gradients, variables), name='train_op')

                pred_x, pred_x_length = self.predict_iterator.get_next()
                pred_logits_intent, pred_logits_ner = network(pred_x, pred_x_length)
                self.prediction = tf.nn.softmax(pred_logits_intent, name='output_intent'), \
                                  tf.nn.softmax(pred_logits_ner, name='output_ner')

                # Tensorboard
                if self.config['tensorboard'] is not None:
                    tf.summary.scalar('cost', self.cost)
                    self.tb_merged = tf.summary.merge_all()

        self.graph = graph
        return graph

    def init_tensorflow(self, graph=None, init_param=True, init_sess=True):
        super().init_tensorflow(graph=graph, init_param=init_param, init_sess=init_sess)

        if init_param:
            # only init embedding when initializing other variables
            self.init_embedding(self.word_embedding)

    def train(self, epochs=400, cancellation_token=None):

        self.sess.run(self.iterator.initializer, feed_dict={self.data_count: len(self.data['train'])})

        epoch = 0

        while epoch < epochs:

            if cancellation_token is not None and cancellation_token.is_cancalled:
                return  # early stopping

            batch_num = 0

            try:
                while batch_num < self.data.steps_per_epoch:

                    if (self.config['display_step'] <= 1 or
                        self.config['epoch'] % self.config['display_step'] == 0 or
                        epoch == epochs) and \
                            (batch_num % 100 == 0 or batch_num == 0):

                        # record cost too

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

        model = IntentNERModel()
        model.restore_config(directory, name)
        if data is not None:
            model.data = data
        else:
            model.data = am.IntentNERData()

        model.build_graph(model.model_config(), model.data)
        model.init_tensorflow(init_param=False, init_sess=True)

        checkpoint = tf.train.get_checkpoint_state(directory)
        input_checkpoint = checkpoint.model_checkpoint_path

        with model.graph.as_default():
            model.saver.restore(model.sess, input_checkpoint)

        model.saved_directory = directory
        model.saved_name = name

        return model

    def predict(self, input_data=None, save_path=None, raw=False):

        if input_data is None:
            input_data = self.data
        elif isinstance(input_data, am.IntentNERData):
            self.data = input_data
        else:
            self.data.set_input(input_data)  # try to match type

        with self.graph.device('/cpu:0'):
            self.sess.run(self.predict_iterator.initializer, feed_dict={self.data_count: len(self.data['input'])})

        outputs_intent = []
        outputs_ner = []
        batch_num = 0
        try:
            while batch_num < self.data.predict_steps:
                intents, ner = self.sess.run(self.prediction)

                outputs_intent.append(intents)
                outputs_ner.append(ner)

                batch_num += 1
        except tf.errors.OutOfRangeError:
            print(batch_num)

        import numpy as np
        outputs_intent = np.concatenate(outputs_intent)
        outputs_ner = np.concatenate(outputs_ner)[:]

        if raw:
            results = list(zip(outputs_intent.tolist(), outputs_ner.tolist()))
        else:
            # give only max
            max_intent = np.argmax(outputs_intent, axis=-1).tolist()
            max_ner = np.argmax(outputs_ner, axis=-1).tolist()

            for i in range(len(max_ner)):
                max_ner[i] = max_ner[i][1:self.data.values['input'][i][1]]  # [0] is <GO>

            results = list(zip(max_intent, max_ner))

        if save_path is not None:
            with open(save_path, "w") as file:
                for i in results:
                    file.write('{0}; {1}\n'.format(str(i[0]), str(i[1])))

        return results
