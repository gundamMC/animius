import tensorflow as tf
from ProjectWaifu.Model import Model
from ProjectWaifu.WordEmbedding import WordEmbedding
from ProjectWaifu.Utils import get_mini_batches, shuffle
import ProjectWaifu.ModelClasses as ModelClasses


class ChatbotModel(Model):

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

    def __init__(self, model_config, data, restore_path=None):

        super().__init__(model_config, data, restore_path=restore_path)

        # Embedding data

        def test_model_structure(key, lambda_value):
            if key in self.model_structure:
                return self.model_structure[key]
            else:
                if self.data is None or 'embedding' not in self.data.values:
                    raise ValueError('When creating a new model, data must contain a word embedding')
                self.model_structure[key] = lambda_value()
                return lambda_value()

        self.n_vector = test_model_structure('n_vector', lambda: len(self.data["embedding"].embedding[0]))
        self.word_count = test_model_structure('word_count', lambda: len(self.data["embedding"].words))

        # just to make it easier to refer to
        self.max_sequence = self.model_structure['max_sequence']

        # Tensorflow placeholders
        self.x = tf.placeholder(tf.int32, [None, self.max_sequence])
        self.x_length = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None, self.max_sequence])
        self.y_length = tf.placeholder(tf.int32, [None])
        self.word_embedding = tf.Variable(tf.constant(0.0, shape=(self.word_count, self.n_vector)), trainable=False)
        self.y_target = tf.placeholder(tf.int32, [None, self.max_sequence])
        # this is w/o <GO>

        # Network parameters
        def get_gru_cell():
            return tf.contrib.rnn.GRUCell(self.model_structure['n_hidden'])

        self.cell_encode = tf.contrib.rnn.MultiRNNCell([get_gru_cell() for _ in range(self.model_structure['layer'])])
        self.cell_decode = tf.contrib.rnn.MultiRNNCell([get_gru_cell() for _ in range(self.model_structure['layer'])])
        self.projection_layer = tf.layers.Dense(self.word_count)

        # Optimization
        dynamic_max_sequence = tf.reduce_max(self.y_length)
        mask = tf.sequence_mask(self.y_length, maxlen=dynamic_max_sequence, dtype=tf.float32)

        # Manual cost
        # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.y_target[:, :dynamic_max_sequence], logits=self.network())
        # self.cost = tf.reduce_sum(crossent * mask) / tf.cast(tf.shape(self.y)[0], tf.float32)

        # Built-in cost
        self.cost = tf.contrib.seq2seq.sequence_loss(self.network(), self.y_target[:, :dynamic_max_sequence], weights=mask)

        optimizer = tf.train.AdamOptimizer(self.hyperparameters['learning_rate'])
        gradients, variables = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, self.model_structure['gradient_clip'])
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        self.infer = self.network(mode="infer")

        # Greedy
        # pred_infer = tf.cond(tf.less(tf.shape(self.infer)[1], self.max_sequence),
        #                      lambda: tf.concat([self.infer,
        #                                         tf.zeros(
        #                                             [tf.shape(self.infer)[0],
        #                                              self.max_sequence - tf.shape(self.infer)[1]],
        #                                             tf.int32)], 1),
        #                      lambda: tf.squeeze(self.infer[:, :20])
        #                      )

        # Beam
        pred_infer = tf.cond(tf.less(tf.shape(self.infer)[2], self.max_sequence),
                             lambda: tf.concat([tf.squeeze(self.infer[:, 0]),
                                                tf.zeros(
                                                    [tf.shape(self.infer)[0],
                                                     self.max_sequence - tf.shape(self.infer)[-1]],
                                                    tf.int32)], 1),
                             lambda: tf.squeeze(self.infer[:, 0, :self.max_sequence])
                             )

        correct_pred = tf.equal(
            pred_infer,
            self.y_target)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Tensorboard
        if self.config['tensorboard'] is not None:
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('accuracy', self.accuracy)
            self.merged = tf.summary.merge_all()

        self.init_tensorflow()

        self.init_hyerdash(self.config['hyperdash'])

        # restore model data values
        self.init_restore(restore_path, self.word_embedding)

    def network(self, mode="train"):

        embedded_x = tf.nn.embedding_lookup(self.word_embedding, self.x)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            self.cell_encode,
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
                    self.cell_decode, attention_mechanism, attention_layer_size=self.model_structure['n_hidden'])
                decoder_initial_state = attn_decoder_cell.zero_state(dtype=tf.float32,
                                                                     batch_size=tf.shape(self.x)[0]
                                                                     ).clone(cell_state=encoder_state)

                embedded_y = tf.nn.embedding_lookup(self.word_embedding, self.y)

                train_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=embedded_y,
                    sequence_length=self.y_length
                )

                # attention
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    attn_decoder_cell,
                    train_helper,
                    decoder_initial_state,
                    output_layer=self.projection_layer
                )

                # decoder = tf.contrib.seq2seq.BasicDecoder(
                #     self.cell_decode,
                #     train_helper,
                #     encoder_state,
                #     output_layer=self.projection_layer
                # )

                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_sequence)

                return outputs.rnn_output
        else:

            with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):

                # Greedy search
                # infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.word_embedding, tf.tile(tf.constant([WordEmbedding.start], dtype=tf.int32), [tf.shape(self.x)[0]]), WordEmbedding.end)
                #
                # decoder = tf.contrib.seq2seq.BasicDecoder(
                #     attn_decoder_cell,
                #     infer_helper,
                #     decoder_initial_state,
                #     output_layer=self.projection_layer
                # )
                #
                # outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_sequence,
                #                                                   impute_finished=True)
                #
                # return outputs.sample_id

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
                    self.cell_decode, attention_mechanism, attention_layer_size=self.model_structure['n_hidden'])

                decoder_initial_state = attn_decoder_cell.zero_state(dtype=tf.float32,
                                                                     batch_size=tf.shape(self.x)[0] * beam_width
                                                                     ).clone(cell_state=encoder_state_beam)

                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=attn_decoder_cell,
                    embedding=self.word_embedding,
                    start_tokens=tf.tile(tf.constant([WordEmbedding.GO], dtype=tf.int32),
                                         [tf.shape(self.x)[0]]),
                    end_token=WordEmbedding.EOS,
                    initial_state=decoder_initial_state,
                    beam_width=beam_width,
                    output_layer=self.projection_layer,
                    length_penalty_weight=0.0
                )

                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_sequence)

                return tf.transpose(outputs.predicted_ids, perm=[0, 2, 1])  # [batch size, beam width, sequence length]

    def train(self, epochs=10):
        for epoch in range(epochs):
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
                summary = self.sess.run(self.merged, feed_dict={
                    self.x: mini_batches_x[0],
                    self.x_length: mini_batches_x_length[0],
                    self.y: mini_batches_y[0],
                    self.y_length: mini_batches_y_length[0],
                    self.y_target: mini_batches_y_target[0]
                })
                self.tensorboard_writer.add_summary(summary, self.config['epoch'])

            self.config['epoch'] += 1

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
            # take only the first beam
            for beam in batch:
                beam_res = ''
                for index in beam:
                    # if test_output is a numpy array, use np.take
                    beam_res = beam_res + input_data['embedding'].words[int(index)] + " "
                result.append(beam_res)
            list_res.append(result)

        if save_path is not None:
            with open(save_path, "w") as file:
                for i in range(len(list_res)):
                    file.write(str(list_res[i][0]) + '\n')

        return list_res


# test

# Creating a model
modelConfig = ModelClasses.ModelConfig(
    config={
        'display_step': 1,
        'tensorboard': './tensorboard',
        'hyperdash': 'Project Waifu Chatbot Model'
    },
    hyperparameters={
        'learning_rate': 0.00015,
        'batch_size': 8,
        'optimizer': 'adam'
    },
    model_structure={
        'max_sequence': 20,
        'n_hidden': 128
    })

data = ModelClasses.ChatbotData(modelConfig)
embedding = WordEmbedding()
embedding.create_embedding("./Data/glove.twitter.27B.100d.txt", vocab_size=40000)

data.add_embedding_class(embedding)

data.add_cornell("./Data/movie_conversations.txt", "./Data/movie_lines.txt", upper_bound=100)
data.add_twitter('./Data/chat.txt', upper_bound=100)

model = ChatbotModel(modelConfig, data)

test = ModelClasses.ChatbotData(modelConfig)
test.add_embedding_class(embedding)
test.parse_input("hello")
test.parse_input("what's your name?")
test.parse_input("fuck you")
test.parse_input("how has your day been?")

model.train(5)
model.save()

model.close()

# restoring the model
# model = ChatbotModel(None, None, restore_path='./model')
#
# embedding = WordEmbedding()
# embedding.create_embedding("./Data/glove.twitter.27B.100d.txt", vocab_size=40000)
# test = ModelClasses.ChatbotData(model.model_structure['max_sequence'])
# test.add_embedding_class(embedding)
# test.parse_input("hello")
#

print(model.predict(test))
