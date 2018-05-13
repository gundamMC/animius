import tensorflow as tf
import numpy as np
from ProjectWaifu.Network import Network
from ProjectWaifu import Utils


class ChatbotNetwork(Network):

    def __init__(self, learning_rate=0.01, batch_size=2048):
        # hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network hyperparameters
        self.n_vector = 50  # word vector size
        self.max_sequence = 30  # maximum number of words allowed
        self.n_hidden = 128  # hidden nodes inside LSTM

        # Tensorflow placeholders
        self.x = tf.placeholder("float", [None, self.max_sequence])
        self.y = tf.placeholder("float", [None, self.max_sequence])

        # Network parameters
        self.cell_encode = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        self.cell_decode = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

    def setTrainingData(self, conv):
        pass

    def network(self, X, mode="train"):
        seqlen = Utils.get_length(X)
        encoder_out, encoder_states = tf.nn.dynamic_rnn(
            self.cell_encode,
            inputs=X,
            dtype=tf.float32,
            sequence_length=seqlen,
            swap_memory=True)

        if mode == "train":
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.y,
                sequence_length=seqlen
            )

        decoder = tf.contrib.seq2seq.BasicDecoder(self.cell_decode,
                                                  helper,
                                                  encoder_states)

        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                       impute_finished=True)
        return outputs

    def train(self, epochs=800, display_step=10):
        pass

    def predict(self, path):
        pass

    def predictAll(self, path, savePath=None):
        pass