import tensorflow as tf
import os
import numpy as np

import sys

from Animius.SpeakerVerification.MFCC import getMFCC


def main(args):

    if len(args) < 1:
        sys.stderr.write(
            'Usage: SpeakerPredict.py <paths to predict>\n')
            
        sys.exit(1)

    paths = []

    for root, directories, file_names in os.walk(args[0]):
        for file in file_names:
            print(os.path.join(root, file))
            paths.append(os.path.join(root, file))

    WaifuGUI = False
    if len(args) > 1 and args[1] == "WaifuGUI":
        WaifuGUI = True

    print('Parameters parsed!')

    # Network hyperparameters
    # Input : 10x39x1

    filter_size_1 = 3
    num_filter_1 = 10
    padding_1 = "SAME"
    # 10x39x10
    
    max_pool_size_1 = 2
    padding_pool = 'SAME'
    # 5x20x10
    
    filter_size_2 = 5
    num_filter_2 = 15
    padding_2 = "SAME"
    # 5x20x15

    fully_connected_1 = 128

    softmax_output = 2

    # placeholders
    x = tf.placeholder(tf.float32, [None, 10, 39, 1])
    y = tf.placeholder(tf.float32, [None, 2])

    weights = {
        # 3x3 conv filter, 1 input layers, 10 output layers
        'wc1': tf.Variable(tf.random_normal([filter_size_1, filter_size_1, 1, num_filter_1])),
        # 5x5 conv filter, 10 input layers, 15 output layers
        'wc2': tf.Variable(tf.random_normal([filter_size_2, filter_size_2, num_filter_1, num_filter_2])),
        # fully connected 1, 15 input layers, 128 outpute nodes
        'wd1': tf.Variable(tf.random_normal([5*20*15, fully_connected_1])),
        # output, 128 input nodes, 2 output nodes
        'out': tf.Variable(tf.random_normal([128, softmax_output]))

        }

    biases = {
        'bc1': tf.Variable(tf.random_normal([num_filter_1])),
        'bc2': tf.Variable(tf.random_normal([num_filter_2])),
        'bd3': tf.Variable(tf.random_normal([fully_connected_1])),
        'out': tf.Variable(tf.random_normal([softmax_output]))
        }

    # Create model
    def conv_net(input_x):
        
        conv1 = tf.nn.conv2d(input_x, weights["wc1"], strides=[1, 1, 1, 1], padding=padding_1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, max_pool_size_1, max_pool_size_1, 1],
                               strides=[1, 2, 2, 1], padding=padding_pool)

        conv2 = tf.nn.conv2d(conv1, weights["wc2"], strides=[1, 1, 1, 1], padding=padding_2)

        fc1 = tf.reshape(conv2, [-1, 5*20*15])
        fc1 = tf.add(tf.matmul(fc1, weights["wd1"]), biases["bd3"])
        fc1 = tf.nn.relu(fc1)

        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

        # softmax is applied during tf.nn.softmax_cross_entropy_with_logits
        return out

    prediction = tf.nn.softmax(conv_net(x))

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Start training
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        saver.restore(sess, "./model/model.ckpt")

        print("\nModel restored")

        file = open("./PredictedClips.waifu", "w")

        for path in paths:
            data = getMFCC(path, False)

            data = data[..., np.newaxis]

            predicted = sess.run(prediction, feed_dict = { x:data })

            result = np.argmax(np.sum(predicted, axis=0) / predicted.shape[0])
            print(path + " - result: " + str(result))
            file.write(path + "-" + str(result) + "\n")

            if WaifuGUI:
                    print("WaifuGUI: " + path + "-" + str(result))


# main
if __name__ == '__main__':
    main(sys.argv[1:])