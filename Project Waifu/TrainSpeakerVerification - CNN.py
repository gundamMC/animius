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
        sys.stderr.write(
            'Usage: SpeakerVerificationTrain.py <learning rate> <drop out keep rate> <epoches> <batch number> <adam beta 1> <adam beta 2> <true paths> <false paths>\n')
        args.append("0.01")
        args.append("0.8")
        args.append("0.01")
        args.append("200")
        args.append("1")
        args.append("0.9")
        args.append("0.999")
        args.append("D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\078 - 折木奉太郎さんですよね 一年B組の.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\079 - 私一年A組なんです.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\083 - はい.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\094 - もうお帰りですか.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\096 - 私戸締りできません.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\160 - でも 後から来た折木さんは鍵が閉まってたと言ってます.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\171 - 気になります 私 なぜ閉じ込められたんでしょう.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\173 - 仮に何かの間違いだというなら 誰のどういう間違いでしょうか.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\176 - 私 気になります.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\192 - 聞こえますよ ほら.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\221 - 入部届けももう書いてるんだし.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\308 - 女郎蜘蛛.wav")
        args.append("D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\001 - 高校生活といえば薔薇色.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\002 - 薔薇色といえば高校生活.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\003 - そう言われるのが当たり前なくらい 高校生活はいつも薔薇色な扱いだよな.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\048 - 神校 FIGHT  FIGHT  FIGHT.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\049 - エネルギー消費の大きい生き方に敬礼.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\055 - ほら これ.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\056 - もう.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\082 - もしかして 音楽の授業で一緒だったか.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\084 - まだ一回しかやってない授業だぞ どんな記憶力だ.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\085 - あ それで千反田さん なぜこの部屋に？.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\092 - 姉貴 喜べ 古典部はめでたく存続したぞ.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\099 - はい.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\149 - しかもその千反田家の長女は成績優秀 眉目秀麗.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\151 - 中学時代県内模試の成績優秀者で よく名前を見かけたよ.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\152 - ほほ そんなに.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\164 - 千反田さんが中から鍵をかけることは不可能ってことだよ.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\193 - 千反田さん 耳いいねえ.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\194 - なんかわかったの 奉太郎.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\235 - （千反田さん それは誤解だよ）.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\252 - 音色に誘われ 音楽室に入ると.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\259 - そう この神山高校にはかつて.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\264 - それにしても意外だね 奉太郎が宿題を忘れて居残りなんて.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\266 - 持ってくるのを忘れたんだ.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\268 - なるほどね.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\271 - 後はこの噂がどう広がっていくか これが重要なんだ.wav,D:\\Project Waifu\\Project-Waifu\\Project Waifu\\chunks\\Hyouka - 01\\307 - 女郎蜘蛛の会.wav")
        

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

    TrainTruePaths = args[7].split(",")
    TrainFalsePaths = args[8].split(",")

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