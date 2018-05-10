import numpy as np


def loadGloveModel(gloveFile):
    printMessage("Loading Glove Model")
    f = open(gloveFile, 'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
        printMessage("Done. " + str(len(model)) + " words loaded!")
    return model


def shuffle(data_lists):
    permutation = list(np.random.permutation(data_lists[0].shape[0]))
    result = []
    for data in data_lists:
        result.append(data[permutation])
    return result


def random_mini_batches(data_lists, mini_batch_number):
    m = data_lists[0].shape[0]
    mini_batches = []

    shuffled = shuffle(data_lists)

    mini_batch_size = np.math.floor(m / mini_batch_number)

    for data in shuffled:
        tmp = []
        for batch in range(0, mini_batch_number):
            tmp.append(data[batch * mini_batch_size: (batch + 1) * mini_batch_size])

        if m % mini_batch_size != 0:
            tmp.append(data[mini_batch_number * mini_batch_size:])

        mini_batches.append(tmp)

    return mini_batches


socket = None


def setSocket(inputSocket):
    global socket
    socket = inputSocket


def printMessage(message):
    if socket is not None:
        socket.send(message.encode("UTF-8"))
    else:
        print(message)
