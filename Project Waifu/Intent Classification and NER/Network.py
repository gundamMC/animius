import tensorflow as tf
import numpy as np


def shuffle(X, Y1, Y2):
    permutation = list(np.random.permutation(X.shape[0]))
    shuffled_X = X[permutation]
    shuffled_Y1 = Y1[permutation]
    shuffled_Y2 = Y2[permutation]
    return shuffled_X, shuffled_Y1, shuffled_Y2


def random_mini_batches(X, Y1, Y2, mini_batch_number):
    m = X.shape[0]
    mini_batches_X = []
    mini_batches_Y1 = []
    mini_batches_Y2 = []

    shuffled_X, shuffled_Y1, shuffled_Y2 = shuffle(X, Y1, Y2)

    mini_batch_size = np.math.floor(m / mini_batch_number)

    for batch in range(0, mini_batch_number):
        mini_batch_X = shuffled_X[batch * mini_batch_size: (batch + 1) * mini_batch_size]
        mini_batch_Y1 = shuffled_Y1[batch * mini_batch_size: (batch + 1) * mini_batch_size]
        mini_batch_Y2 = shuffled_Y2[batch * mini_batch_size: (batch + 1) * mini_batch_size]
        mini_batches_X.append(mini_batch_X)
        mini_batches_Y1.append(mini_batch_Y1)
        mini_batches_Y2.append(mini_batch_Y2)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[mini_batch_number * mini_batch_size:]
        mini_batch_Y1 = shuffled_Y1[mini_batch_number * mini_batch_size:]
        mini_batch_Y2 = shuffled_Y2[mini_batch_number * mini_batch_size:]
        mini_batches_X.append(mini_batch_X)
        mini_batches_Y1.append(mini_batch_Y1)
        mini_batches_Y2.append(mini_batch_Y2)

    return mini_batches_X, mini_batches_Y1, mini_batch_Y2


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


logits = network(x)

prediction = tf.nn.softmax(network(x))

# optimize
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits[0], labels=y_intent)) + \
       tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits[1], labels=y_entities))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)

# initialize variables
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Start training
with tf.Session(config=config) as sess:

    print("start training")
    sess.run(init)

    for epoch in range(epochs + 1):  # since range is exclusive

        mini_batches_X, mini_batches_Y_intent, mini_batches_Y_entities = random_mini_batches(None, None, None)

        for i in range(0, len(mini_batches_X)):

            batch_x = mini_batches_X[i]
            batch_y_intent = mini_batches_Y_intent[i]
            batch_y_entities = mini_batches_Y_entities[i]

            _ = sess.run([optimizer], feed_dict={x: batch_x, y_intent: batch_y_intent, y_entities: batch_y_entities})

            if epoch % display_step == 0:
                cost_value = sess.run([cost],
                                      feed_dict={x: batch_x, y_intent: batch_y_intent, y_entities: batch_y_entities})

                print('epoch', epoch, '- cost', cost_value)

