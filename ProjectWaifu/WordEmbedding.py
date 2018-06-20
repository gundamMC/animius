import numpy as np


words = None
words_to_index = None
embeddings = None

start = 0
end = 0


def create_embedding(glove_path, save_embedding=True, vocab_size=200000):
    global words
    words = []
    global words_to_index
    words_to_index = {}
    global embeddings
    embeddings = []

    print("Loading word vector")

    words_to_index["<UNK>"] = 0
    words.append("<UNK>")

    words_to_index["<GO>"] = 1
    global start
    start = 1
    words.append("<GO>")

    words_to_index["<EOS>"] = 2
    global end
    end = 2
    words.append("<EOS>")

    f = open(glove_path, 'r', encoding='utf8')
    index = 3
    for line in f:

        if index > vocab_size:
            break

        split_line = line.split(' ')
        word = split_line[0]
        if save_embedding:
            embedding = [float(val) for val in split_line[1:]]
            embedding.extend([0, 0, 0])
            embeddings.append(embedding)
        words.append(word)
        words_to_index[word] = index  # 3 special tokens before
        index += 1

    if save_embedding:
        # add special tokens
        zeros = np.zeros((3, 53))
        zeros[0, 50] = 1
        zeros[1, 51] = 1
        zeros[2, 52] = 1

        embeddings = np.array(embeddings)
        embeddings = np.vstack((zeros, embeddings))

        assert embeddings.shape[0] == len(words_to_index)

    print("Word vector loaded")
