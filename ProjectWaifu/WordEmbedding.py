import numpy as np


class WordEmbedding:

    UNK = 0
    GO = 1
    EOS = 2

    def __init__(self):
        self.embedding = None
        self.words = []
        self.words_to_index = {}

        self.words_to_index["<UNK>"] = 0
        self.words.append("<UNK>")

        self.words_to_index["<GO>"] = 1
        self.words.append("<GO>")

        self.words_to_index["<EOS>"] = 2
        self.words.append("<EOS>")

    def create_embedding(self, glove_path, vocab_size=100000):

        self.embedding = []

        f = open(glove_path, 'r', encoding='utf8')
        index = 3
        for line in f:

            if index == vocab_size:
                break

            split_line = line.split(' ')
            word = split_line[0]

            vector = [float(val) for val in split_line[1:]]
            vector.extend([0] * 3)
            self.embedding.append(vector)

            self.words.append(word)
            self.words_to_index[word] = index  # 3 special tokens
            index += 1

        # add special tokens
        vector_length = len(self.embedding[0])
        zeros = np.zeros((3, vector_length))
        zeros[0, vector_length - 3] = 1
        zeros[1, vector_length - 2] = 1
        zeros[2, vector_length - 1] = 1

        self.embedding = np.array(self.embedding)
        self.embedding = np.vstack((zeros, self.embedding))

        assert self.embedding.shape[0] == len(self.words_to_index)
