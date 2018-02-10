import tensorflow as tf
import numpy as np
import math

import sys

import MFCC

np.random.seed(0)

def getData(TruePaths, FalsePaths):
    x0 = np.empty(shape=[0, 10, 39])
    x1 = np.empty(shape=[0, 10, 39])
    for path in TruePaths:
        x0 = np.append(x0, MFCC.getData(path, False), axis = 0)

    for path in FalsePaths:
        x1 = np.append(x1, MFCC.getData(path, False), axis = 0)

    y0 = np.tile([1,0], (x0.shape[0],1))
    y1 = np.tile([0,1], (x1.shape[0],1))

    datax = np.append(x0, x1, axis=0)
    datay = np.append(y0, y1, axis=0)
    
    datax = datax[..., np.newaxis]

    datax, datay = shuffle(datax, datay)

    return datax, datay

def shuffle(X ,Y):
    permutation = list(np.random.permutation(X.shape[0]))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    return shuffled_X, shuffled_Y

def random_mini_batches(X, Y, mini_batch_number):
    m = X.shape[0]
    mini_batches_X = []
    mini_batches_Y = []
    
    shuffled_X, shuffled_Y = shuffle(X,Y)

    mini_batch_size = math.floor(m / mini_batch_number)

    for batch in range(0, mini_batch_number):
        mini_batch_X = shuffled_X[batch * mini_batch_size : (batch + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[batch * mini_batch_size : (batch + 1) * mini_batch_size]
        mini_batches_X.append(mini_batch_X)
        mini_batches_Y.append(mini_batch_Y)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[mini_batch_number * mini_batch_size :]
        mini_batch_Y = shuffled_Y[mini_batch_number * mini_batch_size :]
        mini_batches_X.append(mini_batch_X)
        mini_batches_Y.append(mini_batch_Y)

    return mini_batches_X, mini_batches_Y
        



def main(args):

    if len(args) < 7:
        sys.stderr.write('Usage: SpeakerVerificationTrain.py <learning rate> <drop out keep rate> <epoches> <batch number> <adam beta 1> <adam beta 2> <true paths> <false paths>\n')
        sys.exit(1)

    #args:
    # [0] = Learning rate
    # [1] = drop out keep rate
    # [2] = regulariation lambda
    # [3] = epoches
    # [4] = batch number ("s" for stochastic)
    # [5] = Adam Beta 1
    # [6] = Adam Beta 2
    # [7] = Marked true wavs stored in a text file
    # [8] = Marked false wavs stored in a text file
    # [9] = WaifuGUI


    TrainTruePaths = [line.strip() for line in open(args[7], encoding = 'utf-8')]
    
    TrainFalsePaths = [line.strip() for line in open(args[8], encoding = 'utf-8')]

    train_x, train_y = getData(TrainTruePaths, TrainFalsePaths)

    print(train_x.shape)

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
    if display_epoch < 1:
        display_epoch = 1

    WaifuGUI = False
    if len(args) > 9 and args[9] == "WaifuGUI":
        WaifuGUI = True

    # Network hyperparameters
    #10x39x1

    filter_size_1 = 3
    num_filter_1 = 10
    padding_1 = "SAME"
    #10x39x10
    
    max_pool_size_1 = 2
    padding_pool = 'SAME'
    #5x20x10
    
    filter_size_2 = 5
    num_filter_2 = 15
    padding_2 = "SAME"
    #5x20x15

    fully_connected_1 = 128

    softmax_output = 2

    # placeholders
    x = tf.placeholder(tf.float32, [None, 10, 39, 1])
    y = tf.placeholder(tf.float32, [None, 2])

    weights = {
        # 3x3 conv filter, 1 input layers, 10 output layers
        'wc1' : tf.Variable(tf.random_normal([filter_size_1, filter_size_1, 1, num_filter_1])),
        # 5x5 conv filter, 10 input layers, 15 output layers
        'wc2' : tf.Variable(tf.random_normal([filter_size_2, filter_size_2, num_filter_1, num_filter_2])),
        # fully connected 1, 15 input layers, 128 outpute nodes
        'wd1' : tf.Variable(tf.random_normal([5*20*15, fully_connected_1])),
        # output, 128 input nodes, 2 output nodes
        'out' : tf.Variable(tf.random_normal([128, softmax_output]))

        }

    biases = {
        'bc1' : tf.Variable(tf.random_normal([num_filter_1])),
        'bc2' : tf.Variable(tf.random_normal([num_filter_2])),
        'bd3' : tf.Variable(tf.random_normal([fully_connected_1])),
        'out' : tf.Variable(tf.random_normal([softmax_output]))
        }

    # Create model
    def conv_net(input_x):
        
        conv1 = tf.nn.conv2d(input_x, weights["wc1"], strides = [1,1,1,1], padding = padding_1)
        conv1 = tf.nn.max_pool(conv1, ksize = [1,max_pool_size_1,max_pool_size_1,1], strides = [1,2,2,1], padding = padding_pool)

        conv2 = tf.nn.conv2d(conv1, weights["wc2"], strides = [1,1,1,1], padding = padding_2)

        fc1 = tf.reshape(conv2, [-1, 5*20*15])
        fc1 = tf.add(tf.matmul(fc1, weights["wd1"]), biases["bd3"])
        fc1 = tf.nn.relu(fc1)

        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

        # softmax is applied during tf.nn.softmax_cross_entropy_with_logits
        return out

    prediction = tf.nn.softmax(conv_net(x))

    # optimize

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = conv_net(x), labels = y))

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(conv_net(x), 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialize variables
    init = tf.global_variables_initializer()

    # tensorboard
    tf.summary.scalar("cost", cost)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    #Create a saver object which will save all the variables
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Start training
    with tf.Session(config = config) as sess:
        print('starting training')
        sess.run(init)

        summary_writer = tf.summary.FileWriter('./tmp/', graph=tf.get_default_graph())

        for epoch in range(epoches + 1):

            mini_batches_X, mini_batches_Y = random_mini_batches(train_x, train_y, batch_number)

            for i in range(0, len(mini_batches_X)):
                batch_x = mini_batches_X[i]
                batch_y = mini_batches_Y[i]

                _, summary  = sess.run([train_op, merged_summary_op], feed_dict={x : batch_x, y : batch_y})

                summary_writer.add_summary(summary, epoch * batch_number + i)

            if epoch % display_epoch == 0:
                costprint, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})

                print('epoch', epoch, '- cost', costprint, '- accuracy', acc)

                if WaifuGUI:
                    print("WaifuGUI: " + str(epoch) + "-" + str(costprint)) # accuracy doesn't really show the improvements of the network over epoches as they can reach "satisfying" values before the cost does.

            costprint, acc = sess.run([cost, accuracy], feed_dict={x: train_x, y: train_y})

        saver.save(sess, "./model/model.ckpt")

        print('Done! Cost:', costprint, "Accuracy:", acc)

        if WaifuGUI:
            print("WaifuGUI: " + str(epoches) + "-" + str(costprint)) # accuracy doesn't really show the improvements of the network over epoches as they can reach "satisfying" values before the cost does.


if __name__ == '__main__':
    main(sys.argv[1:])