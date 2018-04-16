import tensorflow as tf
import numpy as np
from ParseData import get_data


def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile, 'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


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

    return mini_batches_X, mini_batches_Y1, mini_batches_Y2


tf.set_random_seed(1)

# hyperparameters
learning_rate = 0.1
epochs = 1000
batch_size = 198
display_step = 50

# Network hyperparameters
n_input = 50            # word vector size
max_sequence = 30       # maximum number of words allowed
n_hidden = 128           # hidden nodes inside LSTM
n_intent_output = 15    # the number of intent classes
n_entities_output = 8   # the number of entity classes (7 classes + 1 none)

# get data
input_data, ner_data, intent_data = get_data()

input_data = input_data.tolist()

glove = loadGloveModel(".\\data\\glove.twitter.27B.50d.txt")

for batch in range(len(input_data)):  # [batch, word]
    for index in range(len(input_data[batch])):
        word = input_data[batch][index].lower()
        if word in glove:
            input_data[batch][index] = glove[word]
        elif word == "<end>":
            input_data[batch][index] = [0] * 50
        else:
            input_data[batch][index] = glove["<unknown>"]

input_data = np.asarray(input_data)


# Tensorflow placeholders
x = tf.placeholder("float", [None, max_sequence, n_input])  # [batch size, sequence length, input length]
y_intent = tf.placeholder("float", [None, n_intent_output])               # [batch size, intent]
y_entities = tf.placeholder("float", [None, max_sequence, n_entities_output])

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
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    # reducing the features to scalars of the maximum
    # and then converting them to "1"s to create a sequence mask
    # i.e. all "sequence length" with "input length" values are converted to a scalar of 1

    length = tf.reduce_sum(used, reduction_indices=1)  # get length by counting how many "1"s there are in the sequence
    length = tf.cast(length, tf.int32)
    return length


def network(X):
    # X has shape of x ([batch size, sequence length, input length])
    cell_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    seqlen = get_length(X)
    input_size = tf.shape(X)[0]

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                 cell_bw,
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
    outputs_intent = tf.matmul(tf.gather_nd(outputs_fw, last_time_step_indexes), weights["out_intent"]) + biases["out_intent"]
    outputs_entities = tf.einsum('ijk,kl->ijl', output_bw, weights["out_entities"]) + biases["out_entities"]

    return outputs_intent, outputs_entities  # linear/no activation as there will be a softmax layer


logits_intent, logits_ner = network(x)

prediction_intent = tf.nn.softmax(logits_intent)
prediction_ner = tf.nn.softmax(logits_ner)

# optimize
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_intent, labels=y_intent)) + \
       tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_ner, labels=y_entities))

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

        mini_batches_X, mini_batches_Y_intent, mini_batches_Y_entities \
            = random_mini_batches(input_data, intent_data, ner_data, int(len(input_data) / batch_size))

        for i in range(0, len(mini_batches_X)):

            batch_x = mini_batches_X[i]
            batch_y_intent = mini_batches_Y_intent[i]
            batch_y_entities = mini_batches_Y_entities[i]

            sess.run(train_op, feed_dict={x: batch_x, y_intent: batch_y_intent, y_entities: batch_y_entities})

            if epoch % display_step == 0:
                cost_value = sess.run([cost],
                                      feed_dict={x: batch_x, y_intent: batch_y_intent, y_entities: batch_y_entities})

                print('epoch', epoch, '- cost', cost_value)

    # print("Testing output:")
    #
    # test = sess.run([tf.argmax(tf.reshape(prediction_ner, [30, 8])[:10], axis=1)],
    #                 feed_dict={x: np.expand_dims(input_data[0], 0)})
    # print(test)
    # print(np.argmax(np.reshape(ner_data[0, :10], [10, 8]), axis=1))
    #
    # test = sess.run([tf.argmax(tf.reshape(prediction_ner, [30, 8])[:7], axis=1)],
    #                 feed_dict={x: np.expand_dims(input_data[1], 0)})
    # print(test)
    # print(np.argmax(np.reshape(ner_data[1, :7], [7, 8]), axis=1))
    #
    # test = sess.run([tf.argmax(tf.reshape(prediction_ner, [30, 8])[:8], axis=1)],
    #                 feed_dict={x: np.expand_dims(input_data[2], 0)})
    # print(test)
    # print(np.argmax(np.reshape(ner_data[2, :8], [8, 8]), axis=1))
