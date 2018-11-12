import numpy as np
import tensorflow as tf


def shuffle(data_lists):
    permutation = list(np.random.permutation(data_lists[0].shape[0]))
    result = []
    for data in data_lists:
        result.append(data[permutation])
    return result


def get_mini_batches(data_lists, mini_batch_size):
    m = data_lists[0].shape[0]
    mini_batches = []

    mini_batch_number = int(m / float(mini_batch_size))

    for data in data_lists:
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


def sentence_to_index(sentence, word_to_index, max_seq=20, go=False, eos=False):
    if go:
        result = [word_to_index["<GO>"]]
        length = 1
    else:
        result = []
        length = 0
    unk = 0
    for word in sentence:
        length += 1
        if word in word_to_index:
            result.append(word_to_index[word])
        else:
            result.append(word_to_index["<UNK>"])
            unk += 1

    if length >= max_seq:
        if eos:
            length = max_seq - 1
        else:
            length = max_seq

    set_sequence_length(result, word_to_index["<EOS>"], max_seq, force_eos=eos)

    return result, length, unk


def set_sequence_length(sequence, pad, max_seq=20, force_eos=False):

    if len(sequence) < max_seq:
        sequence.extend([pad] * (max_seq - len(sequence)))
    elif force_eos:
        sequence = sequence[:max_seq - 1]
        sequence.append(pad)

    return sequence


def setSocket(inputSocket):
    global socket
    socket = inputSocket


def printMessage(message):
    if socket is not None:
        socket.send(message.encode("UTF-8"))
    else:
        print(message)
