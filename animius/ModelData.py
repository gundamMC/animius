import errno
import json
from abc import ABC, abstractmethod
from os import mkdir
from os.path import join

import animius as am
import numpy as np


class Data(ABC):

    def __init__(self):
        self.values = {}
        self.saved_directory = None
        self.saved_name = None
        self.model_config = None

    def set_model_config(self, model_config):
        self.model_config = model_config

    def __getitem__(self, item):
        return self.values[item]

    def add_embedding_class(self, embedding_class):
        self.values["embedding"] = embedding_class

    def __str__(self):
        return str(self.values)

    @abstractmethod
    def add_data(self, data):
        pass

    @abstractmethod
    def parse(self, item):
        pass

    def reset(self):
        self.__init__()

    def save(self, directory=None, name='model_data', save_embedding=False):
        """
        Save a model data object to a directory

        :param directory: directory to save the data
        :param name: string to name the saved files
        :param save_embedding: whether to save a separate copy of the word embedding
        :return: the directory in which the data is saved
        """
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

        shallow_copy = dict(self.values)

        if 'embedding' in self.values:
            # Save it if embedding is not saved or if the user wants to save a separate copy
            if self.values['embedding'].saved_directory is None or save_embedding:
                saved_embedding_directory = join(directory, 'embedding')
                self.values['embedding'].save(saved_embedding_directory, name=name)

            # add embedding values to json
            shallow_copy.pop('embedding')
            shallow_copy['embedding_directory'] = self.values['embedding'].saved_directory
            shallow_copy['embedding_name'] = self.values['embedding'].saved_name

        # dictionary of configs to save as json
        save_dict = {'cls': type(self).__name__,
                     'saved_directory': directory,
                     'save_name': name,
                     'values': shallow_copy}

        self.saved_directory = directory
        self.saved_name = name

        with open(join(directory, name + '.json'), 'w') as f:
            json.dump(save_dict, f, indent=4)

        return directory

    @staticmethod
    def load(directory, name='model_data', console=None):
        """
        Load a model data object from a saved directory

        :param directory: path to the directory in which the model data is saved
        :param name: name of the saved files
        :param console: console object used to check if embeddings and configs have already been loaded in the console
        :return: a model data object
        """
        with open(join(directory, name + '.json'), 'r') as f:
            stored = json.load(f)

        if stored['cls'] == 'ChatData':
            data = ChatData()
        elif stored['cls'] == 'IntentNERData':
            data = IntentNERData()
        elif stored['cls'] == 'SpeakerVerificationData':
            data = SpeakerVerificationData()
        else:
            raise ValueError("Data class not found.")

        # Load data values
        data.values = stored['values']

        if 'embedding_directory' in stored:

            # find if embedding is already loaded in the console
            if console is not None:
                for key, i in console.embeddings.items():
                    if i.saved_directory is not None and \
                            i.saved_directory == stored['embedding_directory'] and \
                            i.saved_name == stored['embedding_name']:

                        if not i.loaded:
                            console.load_embedding(key)

                        data.add_embedding_class(i)
                        break

            # No matching embedding found
            if 'embedding' not in data.values:
                data.add_embedding_class(am.WordEmbedding.load(directory=stored['embedding_directory'],
                                                               name=stored['embedding_name']))

        data.saved_directory = directory
        data.saved_name = name

        return data


class ChatData(Data):

    def __init__(self):

        super().__init__()

        self.values['train_x'] = []
        self.values['train_y'] = []
        self.values['input'] = []

        self.iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count > len(self.values['train_x']):
            raise StopIteration
        else:
            self.iter_count += 1
            return self.values['train_x'][self.iter_count - 1], self.values['train_y'][self.iter_count - 1]

        # iteration object = (train_x sentence, train_y sentence)

    def add_data(self, data):
        assert len(data) == 2
        self.values['train_x'].append(data[0])
        self.values['train_y'].append(data[1])

    def add_input(self, input_x):
        assert isinstance(input_x, str)
        self.values['input'].append(input_x)

    def set_input(self, input_x):
        assert isinstance(input_x, str)
        self.values['input'] = input_x

    def add_files(self, path_x, path_y):

        for line in open(path_x, 'r', encoding='utf8'):
            self.values['train_x'].append(line.lower())

        for line in open(path_y, 'r', encoding='utf8'):
            self.values['train_y'].append(line.lower())

    def add_cornell(self, conversations_path, movie_lines_path, lower_bound=None, upper_bound=None):
        x, y = am.Chatbot.Parse.load_cornell(conversations_path, movie_lines_path)
        self.values['train_x'].extend(x[lower_bound:upper_bound])
        self.values['train_y'].extend(y[lower_bound:upper_bound])

    def add_twitter(self, chat_path, lower_bound=None, upper_bound=None):
        x, y = am.Chatbot.Parse.load_twitter(chat_path)
        self.values['train_x'].extend(x[lower_bound:upper_bound])
        self.values['train_y'].extend(y[lower_bound:upper_bound])

    def parse(self, item):

        if 'embedding' not in self.values:
            raise ValueError('Word embedding not found')

        if isinstance(item, int):
            item = self.values['train_x'][item], self.values['train_y'][item]
            # if item is an index

        item_x, item_y = item  # unpack first

        return am.Chatbot.Parse.data_to_index(item_x, item_y, self.values['embedding'].words_to_index,
                                              max_seq=self.model_config.model_structure['max_sequence'])


class IntentNERData(Data):

    def __init__(self):

        super().__init__()

        self.values['train'] = []
        self.values['input'] = []

    def add_data(self, x_input):
        self.values['train'].append(x_input)

    def add_intent_folder(self, x_path):
        self.values['train'] = am.IntentNER.Parse.get_data(x_path)

    def set_input(self, x_input):
        self.values['input'] = x_input

    def add_input(self, x_input):
        self.values['input'].append(x_input)

    def parse(self, item):
        if 'embedding' not in self.values:
            raise ValueError('Word embedding not found')

        if isinstance(item, int):
            # if item is an index
            input_sentence = self.values['train'][0][item]
            out_intent = self.values['train'][1][item]
            out_ner = self.values['train'][2][item]
        else:
            # unpack
            input_sentence, out_intent, out_ner = item

        input_sentence, input_length, _ = am.Utils.sentence_to_index(input_sentence,
                                                                     word_to_index=self.values[
                                                                         'embedding'].words_to_index,
                                                                     max_seq=self.model_config.model_structure[
                                                                         'max_sequence'],
                                                                     go=True, eos=False)

        return input_sentence, input_length, out_intent, out_ner


class SpeakerVerificationData(Data):

    def __init__(self):

        super().__init__()

        self.values['train_x'] = []
        self.values['train_y'] = []
        self.values['input'] = []

        self.steps_per_epoch_cache = None

    def add_data(self, input_path, is_speaker=True):
        self.values['train_x'].append(input_path)
        self.values['train_y'].append(is_speaker)

    add_wav_file = add_data
    # a simple alias

    def add_text_file(self, input_path, is_speaker=True):
        count = len(self.values['train_x'])
        for line in open(input_path, 'r', encoding='utf8'):
            self.values['train_x'].append(line.strip())

        count = len(self.values['train_x']) - count

        self.values['train_y'].extend([is_speaker] * count)

    def parse(self, item):

        if isinstance(item, np.ndarray):
            item = item[0]

            item = self.values['train_x'][item], self.values['train_y'][item]

        if isinstance(item, int):
            item = self.values['train_x'][item], self.values['train_y'][item]
            # if item is an index

        item_path, item_label = item

        data = am.SpeakerVerification.MFCC.get_MFCC(item_path,
                                                    window=self.model_config.model_structure['input_window'],
                                                    num_cepstral=self.model_config.model_structure['input_cepstral'],
                                                    flatten=False)

        return data, np.repeat(np.array([item_label], dtype='float32'), data.shape[0])

    @property
    def steps_per_epoch(self):
        if self.steps_per_epoch_cache is not None:
            return self.steps_per_epoch_cache
        else:
            total_length = 0
            for item_path in self.values['train_x']:
                # simulate one epoch to obtain steps per epoch
                elements = \
                    am.SpeakerVerification.MFCC.get_MFCC(item_path,
                                                         window=self.model_config.model_structure['input_window'],
                                                         num_cepstral=self.model_config.model_structure['input_cepstral'],
                                                         flatten=False).shape[0]

                total_length += elements

            import math
            self.steps_per_epoch_cache = math.ceil(total_length / 1024)
            return self.steps_per_epoch_cache
