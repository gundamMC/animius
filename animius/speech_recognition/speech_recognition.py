import tensorflow as tf

import animius as am
from animius.Utils import get_mini_batches, shuffle


class SpeechRecognitionModel(am.Model):

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
            'n_hidden': 256,
            'gradient_clip': 5.0,
            'node': 'gru',
            'layer': 1,
            'beam_width': 3,
            'max_sequence': 512,
            'input_cepstral': 39,
            'n_character': 28  # 26 characters + 0th index + blank label
        }

    def __init__(self):

        super().__init__()

        self.x = None
        self.y = None
        self.seq_length = None
        self.pred = None
        self.train_op = None
        self.cost = None
        self.error_rate = None
        self.tb_merged = None

    def build_graph(self, model_config, data, graph=None, embedding_tensor=None):

        # make copies of the dictionaries since we will be editing it
        self.config = dict(model_config.config)
        self.config['class'] = 'SpeechRecognition'
        self.model_structure = dict(model_config.model_structure)
        self.hyperparameters = dict(model_config.hyperparameters)
        self.data = data

        if graph is None:
            graph = tf.Graph()

        # build map
        with graph.as_default():

            if 'GPU' in self.config['device'] and not tf.test.is_gpu_available():
                self.config['device'] = '/cpu:0'
                # override to CPU since no GPU is available

            with graph.device(self.config['device']):

                # Tensorflow placeholders
                self.x = tf.placeholder(tf.int32, [None, None, self.model_structure['input_cepstral']], name='input_x')
                self.y = tf.sparse_placeholder(tf.int32, [None, None], name='train_y')
                self.seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
                # seq length should be the same for both x and y

                # Network parameters
                def get_gru_cell():
                    return tf.contrib.rnn.GRUCell(self.model_structure['n_hidden'])

                # cell_fw = tf.contrib.rnn.MultiRNNCell(
                #     [get_gru_cell() for _ in range(self.model_structure['layer'])])
                # cell_decode = tf.contrib.rnn.MultiRNNCell(
                #     [get_gru_cell() for _ in range(self.model_structure['layer'])])
                # projection_layer = tf.layers.Dense(word_count)

                cell_fw = get_gru_cell()
                cell_bw = get_gru_cell()

                weights = tf.Variable(tf.random_normal([self.model_structure['n_hidden'],
                                                        self.model_structure['n_character']]))

                bias = tf.Variable(tf.random_normal([self.model_structure['n_character']]))

                # Setup model network

                def network():

                    rnn_outputs, _, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw,
                        cell_bw,
                        inputs=self.x,
                        sequence_length=self.seq_length
                    )

                    shape = tf.shape(self.x)
                    batch_size, max_time_steps = shape[0], shape[1]

                    # Reshaping to apply the same weights over the timesteps
                    rnn_outputs = tf.reshape(rnn_outputs[:, :, -1], [-1, self.model_structure['n_hidden']])

                    logits = tf.matmul(rnn_outputs, weights) + bias

                    logits = tf.reshape(logits, [batch_size, -1, self.model_structure['n_character']])

                    return logits

                # Optimization
                self.cost = tf.nn.ctc_loss(self.y, network(), self.seq_length)

                optimizer = tf.train.AdamOptimizer(self.hyperparameters['learning_rate'])
                # gradients, variables = zip(*optimizer.compute_gradients(self.cost))
                # gradients, _ = tf.clip_by_global_norm(gradients, self.model_structure['gradient_clip'])
                # self.train_op = optimizer.apply_gradients(zip(gradients, variables), name='train_op')
                self.train_op = optimizer.minimize(self.cost)

                # Beam
                self.pred = tf.nn.ctc_beam_search_decoder(network(), self.seq_length)[0][0]

                self.error_rate = tf.reduce_sum(tf.edit_distance(self.pred, self.y, normalize=False))

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

            mini_batches_x, mini_batches_y, mini_batches_seq_length \
                = get_mini_batches(
                    shuffle([
                        self.data['x'],
                        self.data['y'],
                        self.data['seq_length']]
                    ),
                    self.hyperparameters['batch_size'])

            for batch in range(len(mini_batches_x)):
                batch_x = mini_batches_x[batch]
                batch_y = mini_batches_y[batch]
                batch_seq_length = mini_batches_seq_length[batch]

                if (self.config['display_step'] == 0 or
                    self.config['epoch'] % self.config['display_step'] == 0 or
                    epoch == epochs) and \
                        (batch % 100 == 0 or batch == 0):
                    _, cost_value = self.sess.run([self.train_op, self.cost], feed_dict={
                        self.x: batch_x,
                        self.y: batch_y,
                        self.seq_length: batch_seq_length
                    })

                    print("epoch:", self.config['epoch'], "- (", batch, "/", len(mini_batches_x), ") -", cost_value)

                    self.config['cost'] = cost_value.item()

                    if self.config['hyperdash'] is not None:
                        self.hyperdash.metric("cost", cost_value)

                else:
                    self.sess.run([self.train_op], feed_dict={
                        self.x: batch_x,
                        self.y: batch_y,
                        self.seq_length: batch_seq_length
                    })

            if self.config['tensorboard'] is not None:
                summary = self.sess.run(self.tb_merged, feed_dict={
                    self.x: mini_batches_x[0],
                    self.y: mini_batches_y[0],
                    self.seq_length: mini_batches_seq_length[0]
                })
                self.tensorboard_writer.add_summary(summary, self.config['epoch'])

            self.config['epoch'] += 1

    @classmethod
    def load(cls, directory, name='model', data=None):

        model = SpeechRecognitionModel()
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
        model.y = model.sess.graph.get_tensor_by_name('train_y:0')
        model.y_length = model.sess.graph.get_tensor_by_name('train_y_length:0')
        model.y_target = model.sess.graph.get_tensor_by_name('train_y_target:0')
        model.train_op = model.sess.graph.get_operation_by_name('train_op')
        model.cost = model.sess.graph.get_tensor_by_name('train_cost/truediv:0')
        model.infer = model.sess.graph.get_tensor_by_name('decode_1/output_infer:0')

        model.init_tensorflow(graph, init_param=False, init_sess=False)

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
