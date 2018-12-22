import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph as tf_freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


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

    result = set_sequence_length(result, word_to_index["<EOS>"], max_seq, force_eos=eos)

    return result, length, unk


def set_sequence_length(sequence, pad, max_seq=20, force_eos=False):

    if len(sequence) < max_seq:
        sequence.extend([pad] * (max_seq - len(sequence)))

    if force_eos:
        sequence = sequence[:max_seq - 1]
        sequence.append(pad)
    else:
        sequence = sequence[:max_seq]

    return sequence


def freeze_graph(model_dir, output_node_names):
    # Retrieve latest checkpoint
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # Define the path for the frozen model
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    input_graph = absolute_model_dir + "/model_graph.pb"
    output_graph = absolute_model_dir + "/frozen_model.pb"

    clear_devices = True

    print(input_graph)

    tf_freeze_graph.freeze_graph(input_graph, None, True,
                                 input_checkpoint, output_node_names,
                                 "", "", output_graph, clear_devices, "",
                                 input_meta_graph=input_checkpoint + ".meta"
                                 )

    return output_graph


def optimize(frozen_graph_path, input_node_names, output_node_names):
    inputGraph = tf.GraphDef()
    with tf.gfile.Open(frozen_graph_path, "rb") as f:
        data2read = f.read()
        inputGraph.ParseFromString(data2read)

    outputGraph = optimize_for_inference_lib.optimize_for_inference(
        inputGraph,
        input_node_names,  # an array of the input node(s)
        output_node_names,  # an array of output nodes
        tf.int32.as_datatype_enum)

    # Save the optimized graph
    absolute_model_dir = "/".join(frozen_graph_path.split('/')[:-1])
    output_graph = absolute_model_dir + '/optimized_model.pb'
    f = tf.gfile.FastGFile(output_graph, "w")
    f.write(outputGraph.SerializeToString())

    return output_graph
