from abc import ABC, abstractmethod

import numpy as np

import animius as am


class Data(ABC):

    def __init__(self):
        self.values = {}

    def __getitem__(self, item):
        return self.values[item]

    def add_embedding_class(self, embedding_class):
        self.values["embedding"] = embedding_class

    def __str__(self):
        return str(self.values)

    @abstractmethod
    def add_data(self, data):
        pass


class ChatbotData(Data):

    def __init__(self, model_config):

        super().__init__()

        if isinstance(model_config, am.ModelConfig):
            max_seq = model_config.model_structure['max_sequence']
        elif isinstance(model_config, int):
            max_seq = model_config
        else:
            raise TypeError('sequence_length must be either an integer or a ModelConfig object')

        self.values['x'] = np.zeros((0, max_seq))
        self.values['x_length'] = np.zeros((0,))
        self.values['y'] = np.zeros((0, max_seq))
        self.values['y_length'] = np.zeros((0,))
        self.values['y_target'] = np.zeros((0, max_seq))

    def add_data(self, data):
        self.add_input_data(data[0], data[1])
        self.add_output_data(data[2], data[3], None if len(data) < 5 else data[4])

    def add_input_data(self, input_data, input_length):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        if not isinstance(input_length, np.ndarray):
            input_length = np.array(input_length)

        self.values['x'] = np.concatenate([self.values['x'], input_data])
        self.values['x_length'] = np.concatenate([self.values['x_length'], input_length])

    def add_output_data(self, output_data, output_length, output_target=None):
        if not isinstance(output_data, np.ndarray):
            output_data = np.array(output_data)
        if not isinstance(output_length, np.ndarray):
            output_length = np.array(output_length)
        if output_target is not None and not isinstance(output_target, np.ndarray):
            output_target = np.array(output_target)

        self.values['y'] = np.concatenate([self.values['y'], output_data])
        self.values['y_length'] = np.concatenate([self.values['y_length'], output_length])
        if output_target is None:
            self.values['y_target'] = np.concatenate([
                self.values['y_target'],
                np.concatenate([
                    output_data[:, 1:],
                    np.full([output_data.shape[0], 1], self.values['embedding'].EOS)], axis=1)
            ])
        else:
            self.values['y_target'] = np.concatenate([self.values['y_target'], np.array(output_target)])

    def add_parse_input(self, input_x):
        x, x_length, _ = am.Utils.sentence_to_index(am.Chatbot.Parse.split_sentence(input_x.lower()),
                                           self.values['embedding'].words_to_index, go=True, eos=True)

        self.add_input_data(np.array(x).reshape(1, len(x)), np.array(x_length).reshape(1, ))

    def set_parse_input(self, input_x):
        x, x_length, _ = am.Utils.sentence_to_index(am.Chatbot.Parse.split_sentence(input_x.lower()),
                                                    self.values['embedding'].words_to_index, go=True, eos=True)

        # directly set the values
        self.values['x'] = np.array(x).reshape((1, len(x)))
        self.values['x_length'] = np.array(x_length).reshape(1, )

    def add_parse_file(self, path):
        x = []
        x_length = []

        f = open(path, 'r', encoding='utf8')
        for line in f:
            x_tmp, length_tmp, _ = am.Utils.sentence_to_index(am.Chatbot.Parse.split_sentence(line.lower()),
                                                     self.values['embedding'].words_to_index, go=True, eos=True)
            x.append(x_tmp)
            x_length.append(length_tmp)

        self.values['x'] = np.concatenate([self.values['x'], np.array(x)])
        self.values['x_length'] = np.concatenate([self.values['x_length'], np.array(x_length)])

    def add_parse_sentence(self, x, y):
        x = am.Chatbot.Parse.split_data(x)
        y = am.Chatbot.Parse.split_data(y)

        x, y, x_length, y_length, y_target = \
            am.Chatbot.Parse.data_to_index(x, y, self.values['embedding'].words_to_index,
                                           max_seq=self.values['x'].shape[-1])

        self.add_data([x, x_length, y, y_length, y_target])

    def add_cornell(self, conversations_path, movie_lines_path, lower_bound=None, upper_bound=None):
        x, y = am.Chatbot.Parse.load_cornell(conversations_path, movie_lines_path)
        self.add_parse_sentence(x[lower_bound:upper_bound], y[lower_bound:upper_bound])

    def add_twitter(self, chat_path, lower_bound=None, upper_bound=None):
        x, y = am.Chatbot.Parse.load_twitter(chat_path)
        self.add_parse_sentence(x[lower_bound:upper_bound], y[lower_bound:upper_bound])


class IntentNERData(Data):

    def __init__(self, model_config):

        super().__init__()

        if isinstance(model_config, am.ModelConfig):
            self.max_seq = model_config.model_structure['max_sequence']
        else:
            raise TypeError('model_config must be a ModelConfig object')

        self.values['x'] = np.zeros((0, self.max_seq))
        self.values['x_length'] = np.zeros((0,))
        self.values['y_intent'] = np.zeros((0, model_config.model_structure['n_intent_output']))
        self.values['y_ner'] = np.zeros((0, self.max_seq, model_config.model_structure['n_ner_output']))

    def add_data(self, data):
        self.add_input_data(data[0], data[1])
        self.add_output_data(data[2], data[3])

    def add_input_data(self, input_data, input_length):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        if not isinstance(input_length, np.ndarray):
            input_length = np.array(input_length)

        self.values['x'] = np.concatenate([self.values['x'], input_data])
        self.values['x_length'] = np.concatenate([self.values['x_length'], input_length])

    def add_output_data(self, output_intent, output_ner):
        if not isinstance(output_intent, np.ndarray):
            output_intent = np.array(output_intent)
        if not isinstance(output_ner, np.ndarray):
            output_ner = np.array(output_ner)

        self.values['y_intent'] = np.concatenate([self.values['y_intent'], output_intent])
        self.values['y_ner'] = np.concatenate([self.values['y_ner'], output_ner])

    def add_parse_data_folder(self, folder_directory):

        x, x_length, y_intent, y_ner = am.IntentNER.Parse.get_data(folder_directory, self.values['embedding'], self.max_seq)

        self.add_data([x, x_length, y_intent, y_ner])

    def add_parse_input(self, input_x):

        x, x_length, _ = am.Utils.sentence_to_index(am.Chatbot.Parse.split_sentence(input_x.lower()),
                                           self.values['embedding'].words_to_index, go=False, eos=False)

        self.add_input_data(np.array(x).reshape((1, len(x))), np.array(x_length).reshape((1, )))

    def set_parse_input(self, input_x):

        x, x_length, _ = am.Utils.sentence_to_index(am.Chatbot.Parse.split_sentence(input_x.lower()),
                                                    self.values['embedding'].words_to_index, go=False, eos=False)

        self.values['x'] = np.array(x).reshape((1, len(x)))
        self.values['x_length'] = np.array(x_length).reshape((1,))


class SpeakerVerificationData(Data):

    def __init__(self, model_config):

        super().__init__()

        self.mfcc_window = model_config.model_structure['input_window']
        self.mfcc_cepstral = model_config.model_structure['input_cepstral']

        self.values['x'] = np.zeros((0,
                                     self.mfcc_window,
                                     self.mfcc_cepstral))
        self.values['y'] = np.zeros((0, 1))

    def add_data(self, data):
        self.add_input_data(data[0])
        self.add_output_data(data[1])

    def add_input_data(self, input_mfcc):
        assert(isinstance(input_mfcc, np.ndarray))  # get_MFCC() returns a numpy array
        self.values['x'] = np.concatenate([self.values['x'], input_mfcc])

    def add_output_data(self, output_label):
        assert(isinstance(output_label, np.ndarray))
        self.values['y'] = np.concatenate([self.values['y'], output_label])

    def add_parse_input_file(self, path):
        data = am.SpeakerVerification.MFCC.get_MFCC(path, window=self.mfcc_window, num_cepstral=self.mfcc_cepstral, flatten=False)
        self.add_input_data(data)
        return data.shape[0]
        # return batch number

    def add_parse_data_paths(self, paths, output=None):

        count = 0

        for path in paths:
            try:
                count += self.add_parse_input_file(path)
            except Exception as e:
                print(e)

        if output is True:
            self.add_output_data(np.expand_dims(np.tile([1], count), -1))
        elif output is False:
            self.add_output_data(np.expand_dims(np.tile([0], count), -1))

    def add_parse_data_file(self, path, output=None, encoding='utf-8'):
        self.add_parse_data_paths([line.strip() for line in open(path, encoding=encoding)], output=output)


# Prediction data for CombinedPredictionModel (use ChatbotData for CombinedChatbotModel)
class CombinedPredictionData(Data):

    def __init__(self, model_config):

        super().__init__()

        if isinstance(model_config, am.ModelConfig):
            self.max_seq = model_config.model_structure['max_sequence']
        elif isinstance(model_config, int):
            self.max_seq = model_config
        else:
            raise TypeError('sequence_length must be either an integer or a ModelConfig object')

        self.values['x'] = np.zeros((0, self.max_seq))
        self.values['x_length'] = np.zeros((0,))

    def add_data(self, data):
        self.add_input_data(data[0], data[1])

    def add_input_data(self, input_data, input_length):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        if not isinstance(input_length, np.ndarray):
            input_length = np.array(input_length)

        self.values['x'] = np.concatenate([self.values['x'], input_data])
        self.values['x_length'] = np.concatenate([self.values['x_length'], input_length])

    def add_parse_input(self, input_x):
        x, x_length, _ = am.Utils.sentence_to_index(am.Chatbot.Parse.split_sentence(input_x.lower()),
                                                    self.values['embedding'].words_to_index, go=False, eos=True)

        self.add_input_data(np.array(x).reshape(1, len(x)), np.array(x_length).reshape(1,))

    def set_parse_input(self, input_x):
        x, x_length, _ = am.Utils.sentence_to_index(am.Chatbot.Parse.split_sentence(input_x.lower()),
                                                    self.values['embedding'].words_to_index, go=False, eos=True)

        # directly set the values
        self.values['x'] = np.array(x).reshape((1, len(x)))
        self.values['x_length'] = np.array(x_length).reshape(1, )

    def chatbot_format(self, index):
        return np.append([self.values['embedding'].GO], self.values['x'][index]), self.values['x_length'][None] + 1

    def get_chatbot_input(self, index):
        if isinstance(index, int):
            index = []

        new_data = CombinedPredictionData(self.max_seq)
        new_data.add_embedding_class(self.values['embedding'])

        for i in index:
            new_data.add_data(self.chatbot_format(i))

        return new_data
