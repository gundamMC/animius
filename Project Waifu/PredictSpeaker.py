import tensorflow as tf
import numpy as np
import sys

import MFCC

# Network hyperparameters
num_input = 390
num_hidden_1 = 200
num_hidden_2 = 100
num_hidden_3 = 50
num_output = 2

# placeholders
x = tf.placeholder(tf.float32, [None, 390])

weights = {
    'h1' : tf.Variable(tf.random_normal([num_input, num_hidden_1])) * tf.sqrt(2/num_input),
    'h2' : tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])) * tf.sqrt(2/num_hidden_1),
    'h3' : tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3])) * tf.sqrt(2/num_hidden_2),
    'output' : tf.Variable(tf.random_normal([num_hidden_3, num_output])) * tf.sqrt(2/num_hidden_3)
    }

biases = {
    'b1' : tf.Variable(tf.random_normal([num_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([num_hidden_2])),
    'b3' : tf.Variable(tf.random_normal([num_hidden_3])),
    'output' : tf.Variable(tf.random_normal([num_output]))
    }

# Create model
def forward_pass(input_x):
    layer_1 = tf.add(tf.matmul(input_x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #layer_1 = tf.nn.dropout(layer_1, keep_prob = keep_prob)
    
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #layer_2 = tf.nn.dropout(layer_2, keep_prob = keep_prob)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    #layer_3 = tf.nn.dropout(layer_3, keep_prob = keep_prob)

    output = tf.add(tf.matmul(layer_3, weights['output']), biases['output'])
    return output

# output to possibility
def softmax(output):
    return tf.nn.softmax(output)


def main(args):

    if len(args) < 1:
        sys.stderr.write(
            'Usage: SpeakerPredict.py <paths to predict>\n')
        sys.exit(1)

    paths = args[0].split(",")

    WaifuGUI = False
    if len(args) > 1 and args[1] == "WaifuGUI":
        WaifuGUI = True

    predict = softmax(forward_pass(x))

    np.set_printoptions(suppress=True)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        saver.restore(sess, "./model/model.ckpt")

        print ("\nModel restored")

        id = ''

        for path in paths:

            data = MFCC.getData(path)

            predictvar = sess.run(predict, feed_dict ={ x: data})
            # not to override predict

            predictvar /= np.sum(predictvar, axis=1, keepdims=True)

            print("Final result:")
            result = np.argmax(np.sum(predictvar, axis=0) / predictvar.shape[0])
            print(result)

            if WaifuGUI:
                print("WaifuGUI: " + path + "-" + str(result))

            # not resetting it for some reason causes problems...
            predictvar = np.zeros((1,1))


if __name__ == '__main__':
    main(sys.argv[1:])