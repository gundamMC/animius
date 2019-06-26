import tensorflow as tf

import animius as am


# force load beam_search_ops, see https://github.com/tensorflow/tensorflow/issues/12927


class ChatbotModel(am.Model):

    # default values
    @staticmethod
    def DEFAULT_HYPERPARAMETERS():
        return {
            'learning_rate': 0.0001,
            'batch_size': 8,
            'optimizer': 'adam'
        }

    @staticmethod
    def DEFAULT_MODEL_STRUCTURE():
        return {
            'max_sequence': 20,
            'n_hidden': 512,
            'gradient_clip': 5.0,
            'node': 'gru',
            'layer': 2,
            'beam_width': 3
        }

    def __init__(self):

        super().__init__()

        self.x = None
        self.x_length = None
        self.y = None
        self.y_length = None
        self.y_target = None
        self.infer = None
        self.train_op = None
        self.cost = None
        self.accuracy = None
        self.tb_merged = None

        self.word_embedding = None
        self.init_word_embedding = False

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
            # result_x, result_y, lengths_x, lengths_y, result_y_target
            return tf.py_func(self.data.parse, [x, False], [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32])

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
        self.config['class'] = 'Chatbot'
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

        # build map
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
                    self.init_word_embedding = True

                else:
                    word_count, n_vector = embedding_tensor.shape
                    self.word_embedding = embedding_tensor
                    self.init_word_embedding = False  # assume the provided tensor already has values

                # just to make it easier to refer to
                max_sequence = self.model_structure['max_sequence']

                # Tensorflow placeholders
                self.x, self.y, self.x_length, self.y_length, self.y_target = self.iterator.get_next()
                self.y_target.set_shape([None, max_sequence])
                self.y_length.set_shape((None,))

                # this is w/o <GO>

                # Network parameters
                def get_gru_cell():
                    return tf.contrib.rnn.GRUCell(self.model_structure['n_hidden'])

                cell_encode = tf.contrib.rnn.MultiRNNCell(
                    [get_gru_cell() for _ in range(self.model_structure['layer'])])
                cell_decode = tf.contrib.rnn.MultiRNNCell(
                    [get_gru_cell() for _ in range(self.model_structure['layer'])])
                projection_layer = tf.layers.Dense(word_count)

                # Setup model network

                def network(x, x_length, mode="train"):

                    x_length.set_shape((None,))

                    embedded_x = tf.nn.embedding_lookup(self.word_embedding, x)
                    embedded_x.set_shape([None, self.model_structure['max_sequence'], n_vector])

                    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                        cell_encode,
                        inputs=embedded_x,
                        dtype=tf.float32,
                        sequence_length=x_length)

                    if mode == "train":

                        with tf.variable_scope('decode'):
                            # attention
                            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                num_units=self.model_structure['n_hidden'], memory=encoder_outputs,
                                memory_sequence_length=x_length)

                            attn_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                                cell_decode, attention_mechanism,
                                attention_layer_size=self.model_structure['n_hidden'])
                            decoder_initial_state = attn_decoder_cell.zero_state(dtype=tf.float32,
                                                                                 batch_size=tf.shape(x)[0]
                                                                                 ).clone(cell_state=encoder_state)

                            embedded_y = tf.nn.embedding_lookup(self.word_embedding, self.y)
                            embedded_y.set_shape([None, self.model_structure['max_sequence'], n_vector])

                            train_helper = tf.contrib.seq2seq.TrainingHelper(
                                inputs=embedded_y,
                                sequence_length=self.y_length
                            )

                            # attention
                            decoder = tf.contrib.seq2seq.BasicDecoder(
                                attn_decoder_cell,
                                train_helper,
                                decoder_initial_state,
                                output_layer=projection_layer
                            )
                            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_sequence)

                            return outputs.rnn_output
                    else:

                        with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
                            # Beam search
                            beam_width = self.model_structure['beam_width']

                            # attention
                            encoder_outputs_beam = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
                            encoder_state_beam = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
                            x_length_beam = tf.contrib.seq2seq.tile_batch(x_length, multiplier=beam_width)

                            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                                num_units=self.model_structure['n_hidden'], memory=encoder_outputs_beam,
                                memory_sequence_length=x_length_beam)

                            attn_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                                cell_decode, attention_mechanism,
                                attention_layer_size=self.model_structure['n_hidden'])

                            decoder_initial_state = attn_decoder_cell.zero_state(
                                dtype=tf.float32,
                                batch_size=tf.shape(x)[0] * beam_width
                            ).clone(cell_state=encoder_state_beam)

                            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                cell=attn_decoder_cell,
                                embedding=self.word_embedding,
                                start_tokens=tf.tile(tf.constant([am.WordEmbedding.GO], dtype=tf.int32),
                                                     [tf.shape(x)[0]]),
                                end_token=am.WordEmbedding.EOS,
                                initial_state=decoder_initial_state,
                                beam_width=beam_width,
                                output_layer=projection_layer,
                                length_penalty_weight=0.0
                            )

                            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_sequence)

                            return tf.transpose(outputs.predicted_ids, perm=[0, 2, 1],
                                                name='output_infer')  # [batch size, beam width, sequence length]

                # Optimization
                dynamic_max_sequence = tf.reduce_max(self.y_length)
                mask = tf.sequence_mask(self.y_length, maxlen=dynamic_max_sequence, dtype=tf.float32)

                # Manual cost
                # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                #     labels=self.y_target[:, :dynamic_max_sequence], logits=self.network())
                # self.cost = tf.reduce_sum(crossent * mask) / tf.cast(tf.shape(self.y)[0], tf.float32)

                # Built-in cost
                self.cost = tf.contrib.seq2seq.sequence_loss(network(self.x, self.x_length),
                                                             self.y_target[:, :dynamic_max_sequence],
                                                             weights=mask,
                                                             name='train_cost')

                optimizer = tf.train.AdamOptimizer(self.hyperparameters['learning_rate'])
                gradients, variables = zip(*optimizer.compute_gradients(self.cost))
                gradients, _ = tf.clip_by_global_norm(gradients, self.model_structure['gradient_clip'])
                self.train_op = optimizer.apply_gradients(zip(gradients, variables), name='train_op')

                pred_x, pred_x_length = self.predict_iterator.get_next()
                self.infer = network(pred_x, pred_x_length, mode="infer")

                # Beam
                pred_infer = tf.cond(tf.less(tf.shape(self.infer)[2], max_sequence),
                                     lambda: tf.concat([tf.squeeze(self.infer[:, 0]),
                                                        tf.zeros(
                                                            [tf.shape(self.infer)[0],
                                                             max_sequence - tf.shape(self.infer)[-1]],
                                                            tf.int32)], 1),
                                     lambda: tf.squeeze(self.infer[:, 0, :max_sequence])
                                     )

                correct_pred = tf.equal(
                    pred_infer,
                    self.y_target)
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                # Tensorboard
                if self.config['tensorboard'] is not None:
                    tf.summary.scalar('cost', self.cost)
                    tf.summary.scalar('accuracy', self.accuracy)
                    self.tb_merged = tf.summary.merge_all(name='tensorboard_merged')

        self.graph = graph

        return graph

    def init_tensorflow(self, graph=None, init_param=True, init_sess=True):
        super().init_tensorflow(graph=graph, init_param=init_param, init_sess=init_sess)

        if self.init_word_embedding:
            super().init_embedding(self.word_embedding)

    def train(self, epochs=10, cancellation_token=None):

        self.sess.run(self.iterator.initializer, feed_dict={self.data_count: len(self.data['train_y'])})

        epoch = 0

        while epoch < epochs:

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
                print(batch_num)

            if self.config['tensorboard'] is not None:
                summary = self.sess.run(self.tb_merged)
                self.tensorboard_writer.add_summary(summary, self.config['epoch'])

            self.config['epoch'] += 1
            epoch += 1

    @classmethod
    def load(cls, directory, name='model', data=None):

        model = ChatbotModel()
        model.restore_config(directory, name)
        if data is not None:
            model.data = data
        else:
            model.data = am.ChatData()

        model.build_graph(model.model_config(), model.data)
        model.init_word_embedding = False  # prevent initializing the word embedding again
        model.init_tensorflow(init_param=False, init_sess=True)

        checkpoint = tf.train.get_checkpoint_state(directory)
        input_checkpoint = checkpoint.model_checkpoint_path

        with model.graph.as_default():
            model.saver.restore(model.sess, input_checkpoint)

        model.saved_directory = directory
        model.saved_name = name

        return model

    def predict(self, input_data, save_path=None, raw=False):

        with self.graph.device('/cpu:0'):
            self.sess.run(self.predict_iterator.initializer, feed_dict={self.data_count: len(self.data['input'])})

        outputs = []
        batch_num = 0
        try:
            while batch_num < self.data.predict_steps:
                outputs.append(self.sess.run(self.infer))
                batch_num += 1
        except tf.errors.OutOfRangeError:
            print(batch_num)

        import numpy as np
        outputs = np.concatenate(outputs)
        # [batch, beam (default 3), sequence]

        # Beam
        sentences = [
            ' '.join(
                [input_data['embedding'].words[index] for index in instance[0]  # only read the first beam
                 if index != input_data['embedding'].EOS and index != input_data['embedding'].GO]  # skip EOS & GO
            )
            for instance in outputs]
        # grab the corresponding words based on indexes from output
        # sentences var is list with shape [batch], each item is a string

        if save_path is not None:
            with open(save_path, "w") as file:
                for sentence in sentences:
                    file.write(sentence + '\n')

        if raw:
            return sentences, outputs
        else:
            return sentences
