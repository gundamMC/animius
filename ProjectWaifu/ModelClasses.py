import ProjectWaifu.Chatbot.ParseData as ChatbotParse
import numpy as np


class ModelConfig:

    def __init__(self, config):
        self.hyperparameters = config['hyperparameters']
        self.model_structure = config['model_structure']
        self.display_step = config['display_step']
        self.tensorboard = config['tensorboard']


class Data:

    def __init__(self):
        self.values = {}

    def add_embedding_class(self, embedding_class):
        self.values["embedding"] = embedding_class

    def __str__(self):
        return str(self.values)


class ChatbotData(Data):

    def add_input_data(self, input_data, input_length):
        assert(isinstance(input_data, np.ndarray))
        assert (isinstance(input_length, np.ndarray))
        self.values['x'] = np.concatenate([self.values['x'], input_data])
        self.values['x_length'] = np.concatenate([self.values['x_length'], input_length])

    def add_output_data(self, output_data, output_length):
        assert (isinstance(output_data, np.ndarray))
        assert (isinstance(output_length, np.ndarray))
        self.values['y'] = np.concatenate([self.values['x'], output_data])
        self.values['y_length'] = np.concatenate([self.values['x_length'], output_length])
        self.values['y_target'] = np.concatenate([output_data[:, 1:], np.full([output_data.shape[0], 1], self.values['embedding'].EOS)], axis=1)

    def parse_sentence_data(self, x, y):
        x = ChatbotParse.split_data(x)
        y = ChatbotParse.split_data(y)

        x, y, x_length, y_length, y_target = \
            ChatbotParse.data_to_index(x, y, self.values['embedding'].words_to_index)

        self.values['x'] = np.array(x)
        self.values['y'] = np.array(y)
        self.values['x_length'] = np.array(x_length)
        self.values['y_length'] = np.array(y_length)
        self.values['y_target'] = np.array(y_target)

    def add_cornell(self, conversations_path, movie_lines_path, lower_bound=None, upper_bound=None):
        x, y = ChatbotParse.load_cornell(conversations_path, movie_lines_path)
        self.parse_sentence_data(x[lower_bound:upper_bound], y[lower_bound:upper_bound])

    def add_twitter(self, chat_path, lower_bound=None, upper_bound=None):
        x, y = ChatbotParse.load_twitter(chat_path)
        self.parse_sentence_data(x[lower_bound:upper_bound], y[lower_bound:upper_bound])
