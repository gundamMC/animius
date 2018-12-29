import numpy as np
from os import mkdir
from os.path import join
import pickle
import errno


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

        self.saved_directory = None
        self.saved_name = None

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

    def save(self, directory=None, name='embedding'):

        if directory is None:
            if self.saved_directory is None:
                raise ValueError("Directory must be provided when saving for the first time")
            else:
                directory = self.saved_directory

        if self.saved_name is not None:
            name = self.saved_name

        try:
            # create directory if it does not already exist
            mkdir(directory)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise exc

        with open(join(directory, name + '_words.pkl'), 'wb') as f:
            # You really shouldn't share pickle files so backward compatibility doesn't matter
            pickle.dump(self.words, f, pickle.HIGHEST_PROTOCOL)
        with open(join(directory, name + '_words_to_index.pkl'), 'wb') as f:
            pickle.dump(self.words_to_index, f, pickle.HIGHEST_PROTOCOL)

        np.save(join(directory, name + '.npy'), self.embedding)

        self.saved_directory = directory
        self.saved_name = name

        return directory

    @classmethod
    def load(cls, directory, name='embedding'):

        embedding = cls()

        with open(join(directory, name + '_words.pkl'), 'rb') as f:
            embedding.words = pickle.load(f)
        with open(join(directory, name + 'words_to_index.pkl'), 'w') as f:
            embedding.words_to_index = pickle.load(f)
        embedding.embedding = np.load(join(directory, name + '.npy'))

        embedding.saved_directory = directory
        embedding.saved_name = name

        return embedding
