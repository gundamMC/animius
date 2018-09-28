import tensorflow as tf
import numpy as np
from ProjectWaifu.Network import Network
import ProjectWaifu.Chatbot.ParseData as ParseData
import ProjectWaifu.WordEmbedding as WordEmbedding
from ProjectWaifu.Utils import get_mini_batches, shuffle
from hyperdash import Experiment

# tmp
import shutil


exp = Experiment("Chatbot")


class ChatbotNetwork(Network):

    def __init__(self, learning_rate=0.001, batch_size=8, restore=False):
        # hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network hyperparameters
        self.n_vector = len(WordEmbedding.embeddings[0])
        self.word_count = len(WordEmbedding.words)
        self.max_sequence = 20
        self.n_hidden = 512
        self.gradient_clip = 5.0

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
            return tf.contrib.rnn.GRUCell(self.n_hidden)

        self.cell_encode = tf.contrib.rnn.MultiRNNCell([get_gru_cell() for _ in range(2)])
        self.cell_decode = tf.contrib.rnn.MultiRNNCell([get_gru_cell() for _ in range(2)])
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

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
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
                             lambda: tf.squeeze(self.infer[:, 0, :20])
                             )

        correct_pred = tf.equal(
            pred_infer,
            self.y_target)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Tensorboard
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

        # Tensorflow initialization
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        if restore:
            self.tensorboard_writer = tf.summary.FileWriter('/tmp')
        else:
            self.tensorboard_writer = tf.summary.FileWriter('/tmp', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        if restore is False:
            embedding_placeholder = tf.placeholder(tf.float32, shape=WordEmbedding.embeddings.shape)
            self.sess.run(self.word_embedding.assign(embedding_placeholder),
                          feed_dict={embedding_placeholder: WordEmbedding.embeddings})

        else:
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./model'))

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
                    num_units=self.n_hidden, memory=encoder_outputs,
                    memory_sequence_length=self.x_length)

                attn_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    self.cell_decode, attention_mechanism, attention_layer_size=self.n_hidden)
                decoder_initial_state = attn_decoder_cell.zero_state(dtype=tf.float32,
                                                                     batch_size=tf.shape(self.x)[0]).clone(cell_state=encoder_state)

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
                beam_width = 3

                # attention
                encoder_outputs_beam = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
                encoder_state_beam = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
                x_length_beam = tf.contrib.seq2seq.tile_batch(self.x_length, multiplier=beam_width)

                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.n_hidden, memory=encoder_outputs_beam,
                    memory_sequence_length=x_length_beam)

                attn_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    self.cell_decode, attention_mechanism, attention_layer_size=self.n_hidden)

                decoder_initial_state = attn_decoder_cell.zero_state(dtype=tf.float32,
                                                                     batch_size=tf.shape(self.x)[0] * beam_width).clone(cell_state=encoder_state_beam)

                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=attn_decoder_cell,
                    embedding=self.word_embedding,
                    start_tokens=tf.tile(tf.constant([WordEmbedding.start], dtype=tf.int32), [tf.shape(self.x)[0]]),
                    end_token=WordEmbedding.end,
                    initial_state=decoder_initial_state,
                    beam_width=beam_width,
                    output_layer=self.projection_layer,
                    length_penalty_weight=0.0
                )

                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_sequence)

                return tf.transpose(outputs.predicted_ids, perm=[0, 2, 1])  # [batch size, beam width, sequence length]

    def setTrainingData(self, train_x, train_y):
        train_x = ParseData.split_data(train_x)
        train_y = ParseData.split_data(train_y)

        train_x, train_y, x_length, y_length, y_target = \
            ParseData.data_to_index(train_x, train_y,
                                    WordEmbedding.words_to_index)

        print("Training data :", len(train_x))

        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.train_x_length = np.array(x_length)
        self.train_y_length = np.array(y_length)
        self.train_y_target = np.array(y_target)

    def train(self, epochs=800, display_step=10, epoch_offset=0):
        for epoch in range(epochs):
            mini_batches_x, mini_batches_x_length, mini_batches_y, mini_batches_y_length, mini_batches_y_target \
                = get_mini_batches(
                shuffle([self.train_x, self.train_x_length, self.train_y, self.train_y_length, self.train_y_target])
                , self.batch_size)

            for batch in range(len(mini_batches_x)):
                batch_x = mini_batches_x[batch]
                batch_x_length = mini_batches_x_length[batch]
                batch_y = mini_batches_y[batch]
                batch_y_length = mini_batches_y_length[batch]
                batch_y_target = mini_batches_y_target[batch]

                if (epoch % display_step == 0 or display_step == 0) and (batch % 100 == 0 or batch == 0):
                    _, cost_value = self.sess.run([self.train_op, self.cost], feed_dict={
                        self.x: batch_x,
                        self.x_length: batch_x_length,
                        self.y: batch_y,
                        self.y_length: batch_y_length,
                        self.y_target: batch_y_target
                    })

                    print("epoch:", epoch_offset + epoch, "- (", batch, "/", len(mini_batches_x), ") -", cost_value)
                    exp.metric("cost", cost_value)

                else:
                    self.sess.run([self.train_op], feed_dict={
                        self.x: batch_x,
                        self.x_length: batch_x_length,
                        self.y: batch_y,
                        self.y_length: batch_y_length,
                        self.y_target: batch_y_target
                    })

            summary = self.sess.run(self.merged, feed_dict={
                self.x: mini_batches_x[0],
                self.x_length: mini_batches_x_length[0],
                self.y: mini_batches_y[0],
                self.y_length: mini_batches_y_length[0],
                self.y_target: mini_batches_y_target[0]
            })

            self.tensorboard_writer.add_summary(summary, epoch_offset + epoch)

    def predict(self, sentence):

        input_x, x_length, _ = ParseData.sentence_to_index(ParseData.split_sentence(sentence.lower()),
                                                           WordEmbedding.words_to_index)

        test_output = self.sess.run(self.infer[0],
                                    feed_dict={
                                        self.x: np.array([input_x]),
                                        self.x_length: np.array([x_length])
                                    })

        # Greedy
        # result = ""
        # for i in range(len(test_output)):
        #     result = result + WordEmbedding.words[int(test_output[i])] + "(" + str(test_output[i]) + ") "
        # return result

        # Beam
        list_res = []
        for index in range(len(test_output)):
            result = ""
            for i in range(len(test_output[index])):
                result = result + WordEmbedding.words[int(test_output[index][i])] + " "
            list_res.append(result)

        return list_res

    def predictAll(self, path, save_path=None):
        pass

    def save(self, step=None, meta=True):
        self.saver.save(self.sess, './model/model', global_step=step, write_meta_graph=meta)


# test
question, response = ParseData.load_cornell("./Data/movie_conversations.txt", "./Data/movie_lines.txt")

question_twitter, response_twitter = ParseData.load_twitter("./Data/chat.txt")

WordEmbedding.create_embedding("./Data/glove.twitter.27B.100d.txt", vocab_size=40000)

test = ChatbotNetwork(learning_rate=0.00015, batch_size=16, restore=True)

test.setTrainingData(question[:50000] + question_twitter[:50000] + question_twitter[200000:250000], response[:50000] + response_twitter[:50000] + response_twitter[200000:250000])

step = 55

while True:

    test.train(5, 1, step)

    step += 5
    test.save(step, step == 0)

    if step % 25 == 0:
        shutil.copy('./model/model-' + str(step) + '.data-00000-of-00001', './backup')

    print(test.predict("hello"))

    print(test.predict("what's your name"))

    print(test.predict("fuck you"))

    print(test.predict("how has your day been?"))

