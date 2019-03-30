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

            with graph.device(self.config['device']):

                if embedding_tensor is None:
                    n_vector = test_model_structure('n_vector', lambda: len(self.data["embedding"].embedding[0]))
                    word_count = test_model_structure('word_count', lambda: len(self.data["embedding"].words))
                    self.word_embedding = tf.Variable(tf.constant(0.0, shape=(word_count, n_vector)),
                                                      trainable=False, name='word_embedding')
                else:
                    self.word_embedding = embedding_tensor

                # Tensorflow placeholders
                self.x = tf.placeholder(tf.int32, [None, self.model_structure['max_sequence']],
                                        name='input_x')  # [batch size, sequence length]
                self.x_length = tf.placeholder(tf.int32, [None], name='input_x_length')
                self.y_intent = tf.placeholder("float", [None, self.model_structure['n_intent_output']],
                                               name='train_y_intent')  # [batch size, intent]
                self.y_ner = tf.placeholder("float", [None, self.model_structure['max_sequence'],
                                                      self.model_structure['n_ner_output']], name='train_y_ner')

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

                def network():

                    embedded_x = tf.nn.embedding_lookup(self.word_embedding, self.x)
                    batch_size = tf.shape(self.x)[0]

                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                 cell_bw,
                                                                 inputs=embedded_x,
                                                                 dtype=tf.float32,
                                                                 sequence_length=self.x_length,
                                                                 swap_memory=True)
                    # see https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
                    # swap_memory is optional.
                    outputs_fw, output_bw = outputs  # outputs is a tuple (output_fw, output_bw)

                    # get last time steps
                    indexes = tf.reshape(tf.range(0, batch_size), [batch_size, 1])
                    last_time_steps = tf.reshape(tf.add(self.x_length, -1), [batch_size, 1])
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
                logits_intent, logits_ner = network()
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
                self.prediction = tf.nn.softmax(logits_intent, name='output_intent'), \
                                  tf.nn.softmax(logits_ner, name='output_ner')

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
            super().init_embedding(self.word_embedding)

    def train(self, epochs=400, CancellationToken=None):

        for epoch in range(epochs):

            if CancellationToken is not None and CancellationToken.is_cancalled:
                return  # early stopping

            mini_batches_x, mini_batches_x_length, mini_batches_y_intent, mini_batches_y_ner \
                = am.Utils.get_mini_batches(
                am.Utils.shuffle([
                    self.data['x'],
                    self.data['x_length'],
                    self.data['y_intent'],
                    self.data['y_ner']
                ]),
                self.hyperparameters['batch_size'])

            for batch in range(len(mini_batches_x)):

                batch_x = mini_batches_x[batch]
                batch_x_length = mini_batches_x_length[batch]
                batch_y_intent = mini_batches_y_intent[batch]
                batch_y_ner = mini_batches_y_ner[batch]

                if (self.config['display_step'] == 0 or
                    self.config['epoch'] % self.config['display_step'] == 0 or
                    epoch == epochs) and \
                        (batch % 100 == 0 or batch == 0):
                    _, cost_value = self.sess.run([self.train_op, self.cost], feed_dict={
                        self.x: batch_x,
                        self.x_length: batch_x_length,
                        self.y_intent: batch_y_intent,
                        self.y_ner: batch_y_ner
                    })

                    print("epoch:", self.config['epoch'], "- (", batch, "/", len(mini_batches_x), ") -", cost_value)

                    self.config['cost'] = cost_value.item()

                    if self.config['hyperdash'] is not None:
                        self.hyperdash.metric("cost", cost_value)

                else:
                    self.sess.run([self.train_op], feed_dict={
                        self.x: batch_x,
                        self.x_length: batch_x_length,
                        self.y_intent: batch_y_intent,
                        self.y_ner: batch_y_ner
                    })

            if self.config['tensorboard'] is not None:
                summary = self.sess.run(self.tb_merged, feed_dict={
                    self.x: mini_batches_x[0],
                    self.x_length: mini_batches_x_length[0],
                    self.y_intent: mini_batches_y_intent[0],
                    self.y_ner: mini_batches_y_ner[0]
                })
                self.tensorboard_writer.add_summary(summary, self.config['epoch'])

            self.config['epoch'] += 1

    @classmethod
    def load(cls, directory, name='model', data=None):

        model = IntentNERModel()
        model.restore_config(directory, name)
        if data is not None:
            model.data = data

        graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        model.sess = tf.Session(config=config, graph=graph)

        checkpoint = tf.train.get_checkpoint_state(directory)
        input_checkpoint = checkpoint.model_checkpoint_path

        with graph.as_default():
            model.saver = tf.train.import_meta_graph(input_checkpoint + '.meta')
            model.saver.restore(model.sess, input_checkpoint)

        # set up self attributes used by other methods
        model.x = model.sess.graph.get_tensor_by_name('input_x:0')
        model.x_length = model.sess.graph.get_tensor_by_name('input_x_length:0')
        model.y_intent = model.sess.graph.get_tensor_by_name('train_y_intent:0')
        model.y_ner = model.sess.graph.get_tensor_by_name('train_y_ner:0')
        model.train_op = model.sess.graph.get_operation_by_name('train_op')
        model.cost = model.sess.graph.get_tensor_by_name('train_cost:0')
        model.prediction = model.sess.graph.get_tensor_by_name('output_intent:0'), \
                           model.sess.graph.get_tensor_by_name('output_ner:0')

        model.init_tensorflow(graph, init_param=False, init_sess=False)

        model.saved_directory = directory
        model.saved_name = name

        return model

    def predict(self, input_data, save_path=None):

        intent, ner = self.sess.run(self.prediction,
                                    feed_dict={
                                        self.x: input_data.values['x'],
                                        self.x_length: input_data.values['x_length']
                                    })

        ner = [ner[i, :int(input_data.values['x_length'][i])] for i in range(len(ner))]

        if save_path is not None:
            with open(save_path, "w") as file:
                for i in range(len(intent)):
                    file.write(str(intent[i]) + ' - ' + ', '.join(str(x) for x in ner[i]) + '\n')

        return intent, ner
