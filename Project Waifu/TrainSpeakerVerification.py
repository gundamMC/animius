import tensorflow as tf
import numpy as np
import math

import sys

import MFCC

def getData(TruePaths, FalsePaths):
    x0 = np.empty(shape=[0, 390])
    x1 = np.empty(shape=[0, 390])
    for path in TruePaths:
        x0 = np.append(x0, MFCC.getData(path), axis = 0)

    for path in FalsePaths:
        x1 = np.append(x1, MFCC.getData(path), axis = 0)

    y0 = np.tile([1,0], (x0.shape[0],1))
    y1 = np.tile([0,1], (x1.shape[0],1))

    datax = np.append(x0, x1, axis=0)
    datay = np.append(y0, y1, axis=0)

    data = np.append(datax, datay, axis=1)

    np.random.shuffle(data)

    return data[:, : 390], data[:, 390 :]

def shuffle(X ,Y):
    permutation = list(np.random.permutation(X.shape[0]))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]
    return shuffled_X, shuffled_Y

def random_mini_batches(X, Y, mini_batch_number):
    m = X.shape[0]
    mini_batches_X = []
    mini_batches_Y = []
    
    shuffled_X, shuffled_Y = shuffle(X,Y)

    mini_batch_size = math.floor(m / mini_batch_number)

    for batch in range(0, mini_batch_number):
        mini_batch_X = shuffled_X[batch * mini_batch_size : (batch + 1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[batch * mini_batch_size : (batch + 1) * mini_batch_size, :]
        mini_batches_X.append(mini_batch_X)
        mini_batches_Y.append(mini_batch_Y)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[mini_batch_number * mini_batch_size :, :]
        mini_batch_Y = shuffled_Y[mini_batch_number * mini_batch_size :, :]
        mini_batches_X.append(mini_batch_X)
        mini_batches_Y.append(mini_batch_Y)

    return mini_batches_X, mini_batches_Y
        



def main(args):

    if len(args) < 7:
        sys.stderr.write(
            'Usage: SpeakerVerificationTrain.py <learning rate> <drop out keep rate> <epoches> <batch size> <adam beta 1> <adam beta 2> <true paths> <false paths>\n')
        sys.exit(1)

    #args:
    # [0] = Learning rate
    # [1] = drop out keep rate
    # [2] = regulariation lambda
    # [3] = epoches
    # [4] = batch number ("s" for stochastic)
    # [5] = Adam Beta 1
    # [6] = Adam Beta 2
    # [7] = Marked true wavs
    # [8] = Marked false wavs
    # [9] = WaifuGUI

    #TrainTruePaths = ['chunk-95.wav', 'chunk-157.wav', 'chunk-102.wav', 'chunk-108.wav', 'chunk-112.wav', 'chunk-113.wav', 'chunk-114.wav', 'chunk-115.wav', 'chunk-120.wav', 'chunk-121.wav', 'chunk-122.wav', 'chunk-130.wav', 'chunk-213.wav', 'chunk-214.wav', 'chunk-233.wav', 'chunk-110.wav', 'chunk-164.wav', 'chunk-169.wav', 'chunk-145.wav', 'chunk-158.wav', 'chunk-163.wav', 'chunk-164.wav', 'chunk-165.wav', 'chunk-168.wav', 'chunk-215.wav', 'chunk-218.wav', 'chunk-186.wav']
    #TrainFalsePaths = ['chunk-00.wav', 'chunk-155.wav', 'chunk-73.wav', 'chunk-74.wav', 'chunk-132.wav', 'chunk-134.wav', 'chunk-175.wav', 'chunk-197.wav', 'chunk-198.wav', 'chunk-241.wav', 'chunk-246.wav', 'chunk-271.wav', 'chunk-282.wav', 'chunk-283.wav', 'chunk-58.wav', 'chunk-59.wav', 'chunk-253.wav', 'chunk-285.wav', 'chunk-139.wav', 'chunk-140.wav', 'chunk-142.wav', 'chunk-144.wav', 'chunk-224.wav', 'chunk-72.wav', 'chunk-71.wav', 'chunk-182.wav', 'chunk-187.wav', 'chunk-170.wav', 'chunk-183.wav', 'chunk-184.wav']

    TrainTruePaths = args[7].split(",")
    TrainFalsePaths = args[8].split(",")

    train_x, train_y = getData(TrainTruePaths, TrainFalsePaths)

    assert train_y.shape[1] == 2

    print('MFCC got')

    # Hyperparameters
    learning_rate = float(args[0])
    dropout_keeprate = float(args[1])
    regularization_rate = float(args[2])
    epoches = int(args[3])

    if args[4] == "batch":
        batch_number = train_x.shape[0]
    else:
        batch_number = int(args[4])

    display_epoch = int(epoches/100) # display 100 steps

    WaifuGUI = False
    if len(args) > 9 and args[9] == "WaifuGUI":
        WaifuGUI = True

    # Network hyperparameters
    num_input = 390
    num_hidden_1 = 200
    num_hidden_2 = 100
    num_hidden_3 = 50
    num_output = 2

    # placeholders
    x = tf.placeholder(tf.float32, [None, 390])
    y = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32, ())

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
    def forward_pass(input_x, keep_prob):
        layer_1 = tf.add(tf.matmul(input_x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.dropout(layer_1, keep_prob = keep_prob)
    
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob = keep_prob)

        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.relu(layer_3)
        layer_3 = tf.nn.dropout(layer_3, keep_prob = keep_prob)

        output = tf.add(tf.matmul(layer_3, weights['output']), biases['output'])
        # softmax is applied during tf.nn.softmax_cross_entropy_with_logits
        return output

    # optimize

    cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = forward_pass(x, keep_prob), labels = y)) +
            regularization_rate * 
            (tf.nn.l2_loss(weights["h1"]) +
            tf.nn.l2_loss(weights["h2"]) +
            tf.nn.l2_loss(weights["h3"])))

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(forward_pass(x, keep_prob), 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialize variables
    init = tf.global_variables_initializer()

    # tensorboard
    tf.summary.scalar("cost", cost)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    #Create a saver object which will save all the variables
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:
        print('starting training')
        sess.run(init)

        summary_writer = tf.summary.FileWriter('./tmp/', graph=tf.get_default_graph())

        for epoch in range(epoches + 1):

            mini_batches_X, mini_batches_Y = random_mini_batches(train_x, train_y, batch_number)

            for i in range(0, len(mini_batches_X)):
                batch_x = mini_batches_X[i]
                batch_y = mini_batches_Y[i]

                _, summary  = sess.run([train_op, merged_summary_op], feed_dict={x : batch_x, y : batch_y, keep_prob : dropout_keeprate})

                summary_writer.add_summary(summary, epoch * batch_number + i)

            if epoch % display_epoch == 0:
                costprint, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob : 1})

                print('epoch', epoch, '- cost', costprint, '- accuracy', acc)

                if WaifuGUI:
                    print("WaifuGUI: " + str(epoch) + "-" + str(costprint)) # accuracy doesn't really show the improvements of the network over epoches as they can reach "satisfying" values before the cost does.

            costprint, acc = sess.run([cost, accuracy], feed_dict={x: train_x, y: train_y, keep_prob : 1})

        saver.save(sess, "./model/model.ckpt")

        print('Done! Cost:', costprint, "Accuracy:", acc)

        if WaifuGUI:
            print("WaifuGUI: " + str(epoches) + "-" + str(costprint)) # accuracy doesn't really show the improvements of the network over epoches as they can reach "satisfying" values before the cost does.


if __name__ == '__main__':
    main(sys.argv[1:])