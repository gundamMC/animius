import tensorflow as tf
import numpy as np

# hyperparameters
learning_rate = 0.01
epochs = 1000
batch_size = 128
display_step = 100

# Network hyperparameters
n_input = 250           # word vector size
max_sequence = 30       # maximum number of words allowed
n_hidden = 64           # hidden nodes inside LSTM
n_intent_output = 15    # the number of intent classes
n_entities_output = 8   # the number of entity classes (7 classes + 1 none)

# Tensorflow placeholders
x = tf.placeholder("float", [None, max_sequence, n_input])  # [batch size, sequence length, input length]
y_intent = tf.placeholder("float", [None, 1])               # [batch size, intent]
y_entities = tf.placeholder("float", [None, max_sequence, n_input])

# Network parameters
weights = {  # LSTM weights are created automatically by tensorflow
    "out_intent": tf.Variable(tf.random_normal([n_hidden, n_intent_output])),
    "out_entities": tf.Variable(tf.random_normal([n_hidden, n_entities_output]))
}

biases = {
    "out_intent": tf.Variable(tf.random_normal([n_intent_output])),
    "out_entities": tf.Variable(tf.random_normal([n_entities_output]))
}


def get_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence)))
    # reducing the features to scalars of the maximum
    # and then converting them to "1"s to create a sequence mask
    # i.e. all "sequence length" with "input length" values are converted to a scalar of 1

    length = tf.reduce_sum(used, 1)  # get length by counting how many "1"s there are in the sequence
    length = tf.cast(length, tf.int32)
    return length


def network(X):
    # X has shape of x ([batch size, sequence length, input length])
    cell_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    initial_state_fw = cell_fw.zero_state(batch_size)
    initial_state_bw = cell_bw.zero_state(batch_size)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=X,
                                                      sequence_length=get_length(X),
                                                      initial_state_fw=initial_state_fw,
                                                      initial_state_bw=initial_state_bw,
                                                      swap_memory=True)
    # see https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
    # swap_memory is optional.

    outputs_fw, output_bw = outputs  # outputs is a tuple (output_fw, output_bw)

    # apply linear
    outputs_intent = tf.matmul(outputs_fw[:, -1, :], weights["out_intent"])  # [:, -1, :] gets the last time step
    outputs_entities = tf.matmul(output_bw, weights["out_entities"])  # not sure if this will work...

    return outputs_intent, outputs_entities  # linear/no activation as there will be a softmax layer
