import numpy as np
import tensorflow as tf


words = []
wordsToIndex = []
embeddings = None


def createEmbedding(gloveFile):
    global words
    words = []
    global wordsToIndex
    wordsToIndex = {}
    global embeddings
    embeddings = []
    printMessage("Loading word vector")
    f = open(gloveFile, 'r', encoding='utf8')
    for index, line in enumerate(f):
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        words.append(word)
        embeddings.append(embedding)
        wordsToIndex[word] = index
    printMessage("Done. " + str(len(words)) + " words loaded!")

    embeddings = np.array(embeddings)

    print(embeddings.shape)

    wordsToIndex["<UKN>"] = len(words)
    words.append("<UKN>")
    wordsToIndex["<EOS>"] = len(words)
    words.append("<EOS>")

    np.vstack((embeddings,
               np.random.uniform(low=-1, high=1, size=embeddings.shape[1]),
               np.zeros(embeddings.shape[1])
               ))

    print(embeddings.shape)


def shuffle(data_lists):
    permutation = list(np.random.permutation(data_lists[0].shape[0]))
    result = []
    for data in data_lists:
        result.append(data[permutation])
    return result


def random_mini_batches(data_lists, mini_batch_size):
    m = data_lists[0].shape[0]
    mini_batches = []

    shuffled = shuffle(data_lists)

    mini_batch_number = np.math.floor(m / float(mini_batch_size))

    for data in shuffled:
        tmp = []
        for batch in range(0, mini_batch_number):
            tmp.append(data[batch * mini_batch_size: (batch + 1) * mini_batch_size])

        if m % mini_batch_size != 0:
            tmp.append(data[mini_batch_number * mini_batch_size:])

        mini_batches.append(tmp)

    return mini_batches


socket = None


def get_length(sequence):
    used = tf.sign(tf.abs(sequence))
    # reducing the features to scalars of the maximum
    # and then converting them to "1"s to create a sequence mask
    # i.e. all "sequence length" with "input length" values are converted to a scalar of 1

    length = tf.reduce_sum(used, reduction_indices=1)  # get length by counting how many "1"s there are in the sequence
    length = tf.cast(length, tf.int32)
    return length


def setSocket(inputSocket):
    global socket
    socket = inputSocket


def printMessage(message):
    if socket is not None:
        socket.send(message.encode("UTF-8"))
    else:
        print(message)
