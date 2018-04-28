import numpy as np


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


glove = loadGloveModel(".\\data\\glove.twitter.27B.50d.txt")


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
