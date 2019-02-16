import tensorflow as tf

import animius as am
from animius.Utils import get_mini_batches, shuffle
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
            if embedding_tensor is None:
                n_vector = test_model_structure('n_vector', lambda: len(self.data["embedding"].embedding[0]))
                word_count = test_model_structure('word_count', lambda: len(self.data["embedding"].words))
                word_embedding = tf.Variable(tf.constant(0.0, shape=(word_count, n_vector)),
                                             trainable=False, name='word_embedding')
            else:
                word_count, n_vector = embedding_tensor.shape
                word_embedding = embedding_tensor

            # just to make it easier to refer to
            max_sequence = self.model_structure['max_sequence']

            # Tensorflow placeholders
            self.x = tf.placeholder(tf.int32, [None, max_sequence], name='input_x')
            self.x_length = tf.placeholder(tf.int32, [None], name='input_x_length')
            self.y = tf.placeholder(tf.int32, [None, max_sequence], name='train_y')
            self.y_length = tf.placeholder(tf.int32, [None], name='train_y_length')
            self.y_target = tf.placeholder(tf.int32, [None, max_sequence], name='train_y_target')

            # this is w/o <GO>

            # Network parameters
            def get_gru_cell():
                return tf.contrib.rnn.GRUCell(self.model_structure['n_hidden'])

            cell_encode = tf.contrib.rnn.MultiRNNCell([get_gru_cell() for _ in range(self.model_structure['layer'])])
            cell_decode = tf.contrib.rnn.MultiRNNCell([get_gru_cell() for _ in range(self.model_structure['layer'])])
            projection_layer = tf.layers.Dense(word_count)

            # Setup model network

            def network(mode="train"):

                embedded_x = tf.nn.embedding_lookup(word_embedding, self.x)

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell_encode,
                    inputs=embedded_x,
                    dtype=tf.float32,
                    sequence_length=self.x_length)

                if mode == "train":

                    with tf.variable_scope('decode'):

                        # attention
                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                            num_units=self.model_structure['n_hidden'], memory=encoder_outputs,
                            memory_sequence_length=self.x_length)

                        attn_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                            cell_decode, attention_mechanism,
                            attention_layer_size=self.model_structure['n_hidden'])
                        decoder_initial_state = attn_decoder_cell.zero_state(dtype=tf.float32,
                                                                             batch_size=tf.shape(self.x)[0]
                                                                             ).clone(cell_state=encoder_state)

                        embedded_y = tf.nn.embedding_lookup(word_embedding, self.y)

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
                        x_length_beam = tf.contrib.seq2seq.tile_batch(self.x_length, multiplier=beam_width)

                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                            num_units=self.model_structure['n_hidden'], memory=encoder_outputs_beam,
                            memory_sequence_length=x_length_beam)

                        attn_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                            cell_decode, attention_mechanism,
                            attention_layer_size=self.model_structure['n_hidden'])

                        decoder_initial_state = attn_decoder_cell.zero_state(
                            dtype=tf.float32,
                            batch_size=tf.shape(self.x)[0] * beam_width
                        ).clone(cell_state=encoder_state_beam)

                        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            cell=attn_decoder_cell,
                            embedding=word_embedding,
                            start_tokens=tf.tile(tf.constant([am.WordEmbedding.GO], dtype=tf.int32),
                                                 [tf.shape(self.x)[0]]),
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
            self.cost = tf.contrib.seq2seq.sequence_loss(network(),
                                                         self.y_target[:, :dynamic_max_sequence],
                                                         weights=mask,
                                                         name='train_cost')

            optimizer = tf.train.AdamOptimizer(self.hyperparameters['learning_rate'])
            gradients, variables = zip(*optimizer.compute_gradients(self.cost))
            gradients, _ = tf.clip_by_global_norm(gradients, self.model_structure['gradient_clip'])
            self.train_op = optimizer.apply_gradients(zip(gradients, variables), name='train_op')

            self.infer = network(mode="infer")

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

    def train(self, epochs=10, CancellationToken=None):
        for epoch in range(epochs):

            if CancellationToken is not None and CancellationToken.is_cancalled:
                return  # early stopping

            mini_batches_x, mini_batches_x_length, mini_batches_y, mini_batches_y_length, mini_batches_y_target \
                = get_mini_batches(
                shuffle([
                    self.data['x'],
                    self.data['x_length'],
                    self.data['y'],
                    self.data['y_length'],
                    self.data['y_target']]),
                self.hyperparameters['batch_size'])

            for batch in range(len(mini_batches_x)):
                batch_x = mini_batches_x[batch]
                batch_x_length = mini_batches_x_length[batch]
                batch_y = mini_batches_y[batch]
                batch_y_length = mini_batches_y_length[batch]
                batch_y_target = mini_batches_y_target[batch]

                if (self.config['display_step'] == 0 or
                    self.config['epoch'] % self.config['display_step'] == 0 or
                    epoch == epochs) and \
                        (batch % 100 == 0 or batch == 0):
                    _, cost_value = self.sess.run([self.train_op, self.cost], feed_dict={
                        self.x: batch_x,
                        self.x_length: batch_x_length,
                        self.y: batch_y,
                        self.y_length: batch_y_length,
                        self.y_target: batch_y_target
                    })

                    print("epoch:", self.config['epoch'], "- (", batch, "/", len(mini_batches_x), ") -", cost_value)

                    self.config['cost'] = cost_value.item()

                    if self.config['hyperdash'] is not None:
                        self.hyperdash.metric("cost", cost_value)

                else:
                    self.sess.run([self.train_op], feed_dict={
                        self.x: batch_x,
                        self.x_length: batch_x_length,
                        self.y: batch_y,
                        self.y_length: batch_y_length,
                        self.y_target: batch_y_target
                    })

            if self.config['tensorboard'] is not None:
                summary = self.sess.run(self.tb_merged, feed_dict={
                    self.x: mini_batches_x[0],
                    self.x_length: mini_batches_x_length[0],
                    self.y: mini_batches_y[0],
                    self.y_length: mini_batches_y_length[0],
                    self.y_target: mini_batches_y_target[0]
                })
                self.tensorboard_writer.add_summary(summary, self.config['epoch'])

            self.config['epoch'] += 1

    @classmethod
    def load(cls, directory, name='model', data=None):

        model = ChatbotModel()
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
        model.x_length = model.sess.graph.get_tensor_by_name('input_x_length:0')
        model.y = model.sess.graph.get_tensor_by_name('train_y:0')
        model.y_length = model.sess.graph.get_tensor_by_name('train_y_length:0')
        model.y_target = model.sess.graph.get_tensor_by_name('train_y_target:0')
        model.train_op = model.sess.graph.get_operation_by_name('train_op')
        model.cost = model.sess.graph.get_tensor_by_name('train_cost/truediv:0')
        model.infer = model.sess.graph.get_tensor_by_name('decode_1/output_infer:0')

        model.init_tensorflow(graph)

        model.saved_directory = directory
        model.saved_name = name

        return model

    def predict(self, input_data, save_path=None):

        test_output = self.sess.run(self.infer,
                                    feed_dict={
                                        self.x: input_data.values['x'],
                                        self.x_length: input_data.values['x_length']
                                    })
        # Beam
        list_res = []
        for batch in test_output:
            result = []
            for beam in batch:  # three branches
                beam_res = ''
                for index in beam:
                    # if test_output is a numpy array, use np.take
                    # append the words into a sentence
                    beam_res = beam_res + input_data['embedding'].words[int(index)] + " "
                result.append(beam_res)
            list_res.append(result)

        if save_path is not None:
            with open(save_path, "w") as file:
                for i in range(len(list_res)):
                    file.write(str(list_res[i][0]) + '\n')

        return list_res
