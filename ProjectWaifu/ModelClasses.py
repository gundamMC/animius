import ProjectWaifu.Chatbot.ParseData as ChatbotParse
import numpy as np


class ModelConfig:
    def __init__(self, config, hyperparameters, model_structure):
        self.config = config
        self.hyperparameters = hyperparameters
        self.model_structure = model_structure


class Data:

    def __init__(self):
        self.values = {}

    def __getitem__(self, item):
        return self.values[item]

    def add_embedding_class(self, embedding_class):
        self.values["embedding"] = embedding_class

    def __str__(self):
        return str(self.values)


class ChatbotData(Data):

    def __init__(self, sequence_length):

        super(ChatbotData, self).__init__()

        if isinstance(sequence_length, ModelConfig) and 'max_sequence' in sequence_length.model_structure:
            max_seq = sequence_length.model_structure['max_sequence']
        elif isinstance(sequence_length, int):
            max_seq = sequence_length
        else:
            raise TypeError('sequence_length must be either an integer or a ModelConfig object')

        self.values['x'] = np.zeros((0, max_seq))
        self.values['x_length'] = np.zeros((0,))
        self.values['y'] = np.zeros((0, max_seq))
        self.values['y_length'] = np.zeros((0,))
        self.values['y_target'] = np.zeros((0, max_seq))

    def add_input_data(self, input_data, input_length):
        assert(isinstance(input_data, np.ndarray))
        assert (isinstance(input_length, np.ndarray))
        self.values['x'] = np.concatenate([self.values['x'], input_data])
        self.values['x_length'] = np.concatenate([self.values['x_length'], input_length])

    def add_output_data(self, output_data, output_length):
        assert (isinstance(output_data, np.ndarray))
        assert (isinstance(output_length, np.ndarray))
        self.values['y'] = np.concatenate([self.values['y'], output_data])
        self.values['y_length'] = np.concatenate([self.values['y_length'], output_length])
        self.values['y_target'] = np.concatenate([
            self.values['y_length'],
            np.concatenate([
                output_data[:, 1:],
                np.full([output_data.shape[0], 1], self.values['embedding'].EOS)], axis=1)
        ])

    def parse_input(self, input_x):
        x, x_length, _ = ChatbotParse.sentence_to_index(ChatbotParse.split_sentence(input_x.lower()),
                                                        self.values['embedding'].words_to_index)

        self.values['x'] = np.concatenate([self.values['x'], np.array(x).reshape((1, len(x)))], axis=0)
        self.values['x_length'] = np.concatenate([self.values['x_length'], np.array(x_length).reshape(1,)], axis=0)

    def parse_input_file(self, path):
        x = []
        x_length = []

        f = open(path, 'r', encoding='utf8')
        for line in f:
            x_tmp, length_tmp, _ = ChatbotParse.sentence_to_index(ChatbotParse.split_sentence(line.lower()),
                                                                  self.values['embedding'].words_to_index)
            x.append(x_tmp)
            x_length.append(length_tmp)

        self.values['x'] = np.concatenate([self.values['x'], np.array(x)])
        self.values['x_length'] = np.concatenate([self.values['x_length'], np.array(x_length)])

    def parse_sentence_data(self, x, y):
        x = ChatbotParse.split_data(x)
        y = ChatbotParse.split_data(y)

        x, y, x_length, y_length, y_target = \
            ChatbotParse.data_to_index(x, y, self.values['embedding'].words_to_index)

        self.values['x'] = np.concatenate([self.values['x'], np.array(x)])
        self.values['y'] = np.concatenate([self.values['y'], np.array(y)])
        self.values['x_length'] = np.concatenate([self.values['x_length'], np.array(x_length)])
        self.values['y_length'] = np.concatenate([self.values['y_length'], np.array(y_length)])
        self.values['y_target'] = np.concatenate([self.values['y_target'], np.array(y_target)])

    def add_cornell(self, conversations_path, movie_lines_path, lower_bound=None, upper_bound=None):
        x, y = ChatbotParse.load_cornell(conversations_path, movie_lines_path)
        self.parse_sentence_data(x[lower_bound:upper_bound], y[lower_bound:upper_bound])

    def add_twitter(self, chat_path, lower_bound=None, upper_bound=None):
        x, y = ChatbotParse.load_twitter(chat_path)
        self.parse_sentence_data(x[lower_bound:upper_bound], y[lower_bound:upper_bound])