import tensorflow as tf
import numpy as np
from ProjectWaifu.IntentNER.ParseData import get_data, sentence_to_vec
from ProjectWaifu import Utils
from ProjectWaifu.Network import Network
import os


class IntentNERNetwork(Network):

    def __init__(self, learning_rate=0.01, batch_size=128):
        # hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network hyperparameters
        self.n_input = 50            # word vector size
        self.max_sequence = 30       # maximum number of words allowed
        self.n_hidden = 128           # hidden nodes inside LSTM
        self.n_intent_output = 15    # the number of intent classes
        self.n_entities_output = 8   # the number of entity classes (7 classes + 1 none)

        # Tensorflow placeholders
        self.x = tf.placeholder("float", [None, self.max_sequence, self.n_input])  # [batch size, sequence length, input length]
        self.y_intent = tf.placeholder("float", [None, self.n_intent_output])               # [batch size, intent]
        self.y_entities = tf.placeholder("float", [None, self.max_sequence, self.n_entities_output])

        # Network parameters
        self.weights = {  # LSTM weights are created automatically by tensorflow
            "out_intent": tf.Variable(tf.random_normal([self.n_hidden, self.n_intent_output])),
            "out_entities": tf.Variable(tf.random_normal([self.n_hidden + self.n_intent_output, self.n_entities_output]))
        }

        self.biases = {
            "out_intent": tf.Variable(tf.random_normal([self.n_intent_output])),
            "out_entities": tf.Variable(tf.random_normal([self.n_entities_output]))
        }

        self.cell_fw = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.cell_bw = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

        # Optimization
        self.logits_intent, self.logits_ner = self.network(self.x)

        self.prediction_intent = tf.nn.softmax(self.logits_intent)
        self.prediction_ner = tf.nn.softmax(self.logits_ner)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_intent, labels=self.y_intent)) + \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_ner, labels=self.y_entities))

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Tensorflow initialization
        self.sess = tf.Session(config=self.config)

        self.sess.run(tf.global_variables_initializer())

        self.glove = None
        self.intents_folder = None

        self.data_set = False

    @staticmethod
    def get_length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        # reducing the features to scalars of the maximum
        # and then converting them to "1"s to create a sequence mask
        # i.e. all "sequence length" with "input length" values are converted to a scalar of 1

        length = tf.reduce_sum(used, reduction_indices=1)  # get length by counting how many "1"s there are in the sequence
        length = tf.cast(length, tf.int32)
        return length

    def network(self, X):
        # X has shape of x ([batch size, sequence length, input length])
        seqlen = self.get_length(X)
        input_size = tf.shape(X)[0]

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,
                                                     self.cell_bw,
                                                     inputs=X,
                                                     dtype=tf.float32,
                                                     sequence_length=seqlen,
                                                     swap_memory=True)
        # see https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
        # swap_memory is optional.

        outputs_fw, output_bw = outputs  # outputs is a tuple (output_fw, output_bw)

        # get last time steps
        indexes = tf.reshape(tf.range(0, input_size), [input_size, 1])
        last_time_steps = tf.reshape(tf.add(seqlen, -1), [input_size, 1])
        last_time_step_indexes = tf.concat([indexes, last_time_steps], axis=1)

        # apply linear
        outputs_intent = tf.matmul(tf.gather_nd(outputs_fw, last_time_step_indexes), self.weights["out_intent"]) + self.biases["out_intent"]

        entities = tf.concat([output_bw, tf.tile(tf.expand_dims(outputs_intent, 1), [1, 30, 1])], -1)
        outputs_entities = tf.einsum('ijk,kl->ijl', entities, self.weights["out_entities"]) + self.biases["out_entities"]

        return outputs_intent, outputs_entities  # linear/no activation as there will be a softmax layer

    def train(self, epochs=100, display_step=10):
        if not self.data_set:
            Utils.printMessage("Error: Training data not set")
            return

        # get data
        input_data, ner_data, intent_data = get_data(self.glove, self.intents_folder)

        # Start training
        Utils.printMessage("Starting training")
        for epoch in range(epochs + 1):  # since range is exclusive

            mini_batches_X, mini_batches_Y_intent, mini_batches_Y_entities \
                = Utils.random_mini_batches([input_data, intent_data, ner_data], int(len(input_data) / self.batch_size))

            for i in range(0, len(mini_batches_X)):

                batch_x = mini_batches_X[i]
                batch_y_intent = mini_batches_Y_intent[i]
                batch_y_entities = mini_batches_Y_entities[i]

                self.sess.run(self.train_op, feed_dict={self.x: batch_x,
                                                                self.y_intent: batch_y_intent,
                                                                self.y_entities: batch_y_entities})

                if epoch % display_step == 0:
                    cost_value = self.sess.run([self.cost],
                                               feed_dict={self.x: batch_x,
                                                          self.y_intent: batch_y_intent,
                                                          self.y_entities: batch_y_entities})

                    Utils.printMessage(
                        'epoch ' + str(epoch) + ' (' + str(i + 1) + '/' + str(mini_batches_X) + ') - cost' + str(
                            cost_value))

    def predict(self, sentence):

        response_data = sentence_to_vec(self.glove, sentence.lower().split())

        intent, ner = self.sess.run([tf.argmax(tf.reshape(self.prediction_intent, [self.n_intent_output])),
                                     tf.argmax(tf.reshape(self.prediction_ner, [self.max_sequence,
                                                                                self.n_entities_output]), axis=1)
                                     [:len(str.split(sentence))]],
                                    feed_dict={self.x: np.expand_dims(response_data, 0)})

        return intent, ner

    def predictAll(self, path, savePath=None):
        result = []

        if os.path.isdir(path):
            paths = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    paths.append(os.path.join(root, file))
        else:
            with open(path) as file:
                paths = file.read().splitlines()

        for path in paths:
            result.append(self.predict(path))

        if savePath is not None:
            with open(savePath, "a") as file:
                for i in range(len(paths)):
                    file.write(str(result[i][0]) + " " + str(result[i][1]) + " " + paths[i] + "\n")

        return result

    def setTrainingData(self, intents_folder, glove):
        self.glove = Utils.loadGloveModel(glove)
        self.intents_folder = intents_folder
