import errno
import json
import math
from abc import ABC, abstractmethod
import os

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
            os.mkdir(directory)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise exc

        shallow_copy = dict(self.values)

        if 'embedding' in self.values:
            # Save it if embedding is not saved or if the user wants to save a separate copy
            if self.values['embedding'].saved_directory is None or save_embedding:
                saved_embedding_directory = os.path.join(directory, 'embedding')
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

        with open(os.path.join(directory, name + '.json'), 'w') as f:
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
        with open(os.path.join(directory, name + '.json'), 'r') as f:
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

        self.enable_cache = True
        self.cache = dict()
        self.predict_cache = dict()

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
        if isinstance(input_x, str):
            self.values['input'].append(input_x)
        elif isinstance(input_x, list):
            self.values['input'].extend(input_x)
        elif isinstance(input_x, np.ndarray):
            self.values['input'].extend(input_x.tolist())
        else:
            # try to convert to a list
            self.values['input'].extend(list(input_x))

    def set_input(self, input_x):
        if isinstance(input_x, str):
            self.values['input'] = [input_x]
        elif isinstance(input_x, list):
            self.values['input'] = input_x
        elif isinstance(input_x, np.ndarray):
            self.values['input'] = input_x.tolist()
        else:
            # try to convert to a list
            self.values['input'] = list(input_x)

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

    def parse(self, item, from_input=False):

        if 'embedding' not in self.values:
            raise ValueError('Word embedding not found')

        if isinstance(item, np.ndarray):
            item = int(item[0])

        if from_input:
            if self.enable_cache and item in self.predict_cache:
                return self.predict_cache[item]

            x, x_length, _ = am.Utils.sentence_to_index(self.values['input'][item],
                                                        self.values['embedding'].words_to_index,
                                                        max_seq=self.model_config.model_structure['max_sequence'],
                                                        go=True,
                                                        eos=True)
            if self.enable_cache:
                # turn into int32 first (see issue #3)
                self.cache[item] = np.array(x, np.int32), np.array(x_length, np.int32)
            return np.array(x, np.int32), np.array(x_length, np.int32)

        if self.enable_cache and item in self.cache:
            return self.cache[item]

        if isinstance(item, int):
            item_x = self.values['train_x'][item]
            item_y = self.values['train_y'][item]
            # if item is an index
        else:
            item_x, item_y = item  # try to unpack

        # result_x, result_y, lengths_x, lengths_y, result_y_target
        result = \
            am.Chatbot.Parse.data_to_index(item_x,
                                           item_y,
                                           self.values['embedding'].words_to_index,
                                           max_seq=self.model_config.model_structure['max_sequence'])

        # cast to int32 (as py defaults to int64 on certain platforms)
        result = (np.array(x, np.int32) for x in result)

        if self.enable_cache:
            self.cache[item] = result
            return self.cache[item]
        else:
            return result

    @property
    def steps_per_epoch(self):
        return math.ceil(len(self.values['train_x']) / self.model_config.hyperparameters['batch_size'])

    @property
    def predict_steps(self):
        return math.ceil(len(self.values['input']) / self.model_config.hyperparameters['batch_size'])


class IntentNERData(Data):

    def __init__(self):

        super().__init__()

        self.values['train'] = []
        self.values['input'] = []

        self.folder_tmp = None

    def add_data(self, x_input):
        self.values['train'].append(x_input)

    def set_intent_folder(self, x_path):

        if 'embedding' not in self.values:
            raise ValueError('Word embedding not found')

        if self.model_config is None:
            # no model config yet, wait for a model config to be added
            self.folder_tmp = x_path
            return

        data = am.IntentNER.Parse.get_data(x_path)

        results = []

        for i in range(len(data[0])):
            input_sentence = data[0][i]
            out_intent = data[1][i]
            out_ner = data[2][i]

            input_sentence, input_length, _ = am.Utils.sentence_to_index(input_sentence,
                                                                         word_to_index=self.values[
                                                                             'embedding'].words_to_index,
                                                                         max_seq=self.model_config.model_structure[
                                                                             'max_sequence'],
                                                                         go=True, eos=False)

            out_ner.extend([0] * (self.model_config.model_structure['max_sequence'] - len(out_ner)))

            # cast to int32 (since python use int64 as default on specific platforms)
            results.append((np.array(input_sentence, np.int32), np.array(input_length, np.int32),
                            np.array(out_intent, np.int32), np.array(out_ner, np.int32)))

        self.values['train'] = results

    def set_model_config(self, model_config):
        super().set_model_config(model_config)

        if self.folder_tmp is not None:
            # should be called when building graph
            self.set_intent_folder(self.folder_tmp)
            self.folder_tmp = None

    def add_input(self, input_x):
        if isinstance(input_x, str):
            self.values['input'].append(input_x)
        elif isinstance(input_x, list):
            self.values['input'].extend(input_x)
        elif isinstance(input_x, np.ndarray):
            self.values['input'].extend(input_x.tolist())
        else:
            # try to convert to a list
            self.values['input'].extend(list(input_x))

    def set_input(self, input_x):
        if isinstance(input_x, str):
            self.values['input'] = [input_x]
        elif isinstance(input_x, list):
            self.values['input'] = input_x
        elif isinstance(input_x, np.ndarray):
            self.values['input'] = input_x.tolist()
        else:
            # try to convert to a list
            self.values['input'] = list(input_x)

    def parse(self, item, from_input=False):
        if isinstance(item, np.ndarray):
            if from_input:
                # prediction data
                if isinstance(self.values['input'][item[0]], tuple):
                    return self.values['input'][item[0]]
                else:

                    sentence = str.split(str.lower(self.values['input'][item[0]]))

                    input_sentence, input_length, _ = am.Utils.sentence_to_index(sentence,
                                                                                 word_to_index=self.values[
                                                                                     'embedding'].words_to_index,
                                                                                 max_seq=self.model_config.model_structure[
                                                                                     'max_sequence'],
                                                                                 go=True, eos=False)

                    # cast to int32 (since python use int64 as default on specific platforms)
                    self.values['input'][item[0]] = np.array(input_sentence, np.int32), np.array(input_length, np.int32)
                    return self.values['input'][item[0]]

            # training data
            return self.values['train'][item[0]]
        elif isinstance(item, int):
            if from_input:
                return self.values['input'][item]
            return self.values['train'][item]
        else:
            raise NotImplementedError('Animius currently pre-processes intent NER data for better performance,'
                                      'please input an index instead')

    @property
    def steps_per_epoch(self):
        return math.ceil(len(self.values['train']) / self.model_config.hyperparameters['batch_size'])

    @property
    def predict_steps(self):
        return math.ceil(len(self.values['input']) / self.model_config.hyperparameters['batch_size'])


class SpeakerVerificationData(Data):

    def __init__(self):

        super().__init__()

        self.values['train_x'] = []
        self.values['train_y'] = []
        self.values['input'] = []

        self.steps_per_epoch_cache = None
        self.predict_steps_cache = None
        self.predict_step_nums = dict()

        # brut-force cache to prevent io bottleneck
        self.enable_cache = True
        self.cache = dict()
        self.predict_cache = dict()

    def add_wav_file(self, input_path, is_speaker=True):

        if isinstance(input_path, str):
            input_path = [input_path]

        if is_speaker is None:
            self.values['input'].extend(input_path)
            self.predict_steps_cache = None
            return

        self.values['train_x'].extend(input_path)
        self.values['train_y'].extend(is_speaker)

        self.steps_per_epoch_cache = None  # refresh cache every time the data is modified

    def set_wav_file(self, input_path, is_speaker=True):

        if isinstance(input_path, str):
            input_path = [input_path]

        if is_speaker is None:
            self.values['input'] = input_path
            self.predict_steps_cache = None
            return

        self.values['train_x'] = input_path
        self.values['train_y'] = is_speaker

        self.steps_per_epoch_cache = None  # refresh cache every time the data is modified

    def add_text_file(self, input_path, is_speaker=True):

        if is_speaker is None:
            for line in open(input_path, 'r', encoding='utf8'):
                self.values['input'].append(line.strip())
            self.predict_steps_cache = None
            return

        count = len(self.values['train_x'])
        for line in open(input_path, 'r', encoding='utf8'):
            self.values['train_x'].append(line.strip())

        count = len(self.values['train_x']) - count

        self.values['train_y'].extend([is_speaker] * count)

        self.steps_per_epoch_cache = None

    def set_text_file(self, input_path, is_speaker=True):

        if is_speaker is None:
            res = []
            for line in open(input_path, 'r', encoding='utf8'):
                res.append(line.strip())
            self.values['input'] = res
            self.predict_steps_cache = None
            return

        res = []
        for line in open(input_path, 'r', encoding='utf8'):
            res.append(line.strip())

        self.values['train_x'] = res
        count = len(self.values['train_x'])
        self.values['train_y'] = [is_speaker] * count

        self.steps_per_epoch_cache = None

    def add_folder(self, folder_path, is_speaker=True):
        if is_speaker is None:
            for item in os.scandir(folder_path):
                if item.is_file():
                    self.values['input'].append(item.path)
            self.predict_steps_cache = None
        else:
            count = len(self.values['train_x'])
            for item in os.scandir(folder_path):
                if item.is_file():
                    self.values['train_x'].append(item.path)

            count = len(self.values['train_x']) - count
            self.values['train_y'].extend([is_speaker] * count)

            self.steps_per_epoch_cache = None

    def set_folder(self, folder_path, is_speaker=True):
        if is_speaker is None:
            res = []
            for item in os.scandir(folder_path):
                if item.is_file():
                    res.append(item.path)

            self.values['input'] = res
            self.predict_steps_cache = None
        else:
            res = []
            for item in os.scandir(folder_path):
                if item.is_file():
                    res.append(item.path)

            self.values['train_x'] = res
            count = len(self.values['train_x'])
            self.values['train_y'] = [is_speaker] * count

            self.steps_per_epoch_cache = None

    def parse(self, item, from_input=False):
        if isinstance(item, np.ndarray):
            item = int(item[0])

        # prediction
        if from_input:
            if self.enable_cache and item in self.predict_cache:
                return self.predict_cache[item]

            item_path = self.values['input'][item]
            data = am.audio.get_MFCC(item_path,
                                     window=self.model_config.model_structure['input_window'],
                                     num_cepstral=self.model_config.model_structure['input_cepstral'],
                                     conv=True)

            if self.enable_cache:
                self.cache[item] = data
            return data

        # else, training set
        if self.enable_cache and item in self.cache:
            return self.cache[item]

        # not in cache or cache not enabled, proceed to process
        if isinstance(item, int):
            item_path, item_label = self.values['train_x'][item], self.values['train_y'][item]
            # if item is an index

        else:
            item_path, item_label = item

        data = am.audio.get_MFCC(item_path,
                                 window=self.model_config.model_structure['input_window'],
                                 num_cepstral=self.model_config.model_structure['input_cepstral'],
                                 conv=True)

        if self.enable_cache:
            self.cache[item] = data, np.repeat(np.array([item_label], dtype='float32'), data.shape[0])
            return self.cache[item]
        else:
            return data, np.repeat(np.array([item_label], dtype='float32'), data.shape[0])

    @property
    def steps_per_epoch(self):
        if self.steps_per_epoch_cache is not None:
            return self.steps_per_epoch_cache
        else:
            total_length = 0
            for index in range(len(self.values['train_x'])):
                # simulate one epoch to obtain steps per epoch
                data = \
                    am.audio.get_MFCC(self.values['train_x'][index],
                                                         window=self.model_config.model_structure['input_window'],
                                                         num_cepstral=self.model_config.model_structure['input_cepstral'],
                                                         conv=True)

                elements = data.shape[0]

                # count elements
                total_length += elements

                if self.enable_cache:
                    self.cache[index] = data, np.repeat(
                        np.array([self.values['train_y'][index]], dtype='float32'), elements
                    )

            self.steps_per_epoch_cache = math.ceil(total_length / self.model_config.hyperparameters['batch_size'])
            return self.steps_per_epoch_cache

    @property
    def predict_steps(self):
        if self.predict_steps_cache is not None:
            return self.predict_steps_cache
        else:
            total_length = 0
            for index in range(len(self.values['input'])):
                # simulate one epoch to obtain steps per epoch
                data = \
                    am.audio.get_MFCC(self.values['input'][index],
                                      window=self.model_config.model_structure['input_window'],
                                      num_cepstral=self.model_config.model_structure['input_cepstral'],
                                      conv=True)

                # count elements
                elements = data.shape[0]
                total_length += elements
                self.predict_step_nums[index] = elements

                if self.enable_cache:
                    self.predict_cache[index] = data

            self.predict_steps_cache = math.ceil(total_length / self.model_config.hyperparameters['batch_size'])
            return self.predict_steps_cache


class SpeechRecognitionData(Data):
    def __init__(self, model_config):

        super().__init__(model_config)

        self.model_config = model_config

        self.mfcc_cepstral = self.model_config.model_structure['input_cepstral']

        max_seq = self.model_config.model_structure['max_sequence']

        self.values['x'] = np.zeros((0, max_seq, self.mfcc_cepstral))
        self.values['seq_length'] = np.zeros((0,))
        self.values['y'] = np.zeros((0, max_seq))

    def add_data(self, data):
        self.add_input_data(data[0], data[1])
        self.add_output_data(data[2])

    def add_input_data(self, input_data, input_length):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        if not isinstance(input_length, np.ndarray):
            input_length = np.array(input_length)

        self.values['x'] = np.concatenate([self.values['x'], input_data])
        self.values['seq_length'] = np.concatenate([self.values['seq_length'], input_length])

    def add_output_data(self, output_data):
        if not isinstance(output_data, np.ndarray):
            output_data = np.array(output_data)

        self.values['y'] = np.concatenate([self.values['y'], output_data])

    def add_libri_speech(self, path, sample_rate=None, init=True):
        # this takes way too long...
        # really need to move to tf.data

        input_sequence = []
        sequence_length = []
        output_characters = []

        for dir_name in os.listdir(path):
            # reader dir
            dir_path = os.path.join(path, dir_name)
            if os.path.isdir(dir_path):
                for chapter_id in os.listdir(dir_path):
                    # chapter dir
                    chapter_dir = os.path.join(dir_path, chapter_id)

                    trans_file = None
                    audio_files = {}

                    for item_name in os.listdir(chapter_dir):
                        if item_name.endswith('trans.txt'):
                            trans_file = os.path.join(chapter_dir, item_name)
                        elif item_name.endswith('.flac'):
                            audio_files[item_name.replace('.flac', '')] = os.path.join(chapter_dir, item_name)

                    if trans_file is not None and len(audio_files) > 0:
                        # this folder actually has stuff

                        lines = [line.rstrip('\n') for line in open(trans_file, 'r')]

                        for line in lines:
                            audio_id, trans = line.split(' ', 1)

                            sequence = am.audio.get_MFCC(audio_files[audio_id],
                                                         sample_rate=sample_rate,
                                                         num_cepstral=self.mfcc_cepstral,
                                                         conv=False)

                            input_sequence.append(sequence)
                            sequence_length.append(sequence.shape[0])
                            output_characters.append(list(trans))

        if init:
            max_seq = self.model_config.model_structure['max_sequence'] = max(sequence_length)

            # please don't actually use this
            # move to tf.data
            length_x = len(sequence_length)
            self.values['x'] = np.zeros((length_x, max_seq, self.mfcc_cepstral))
            for i in range(length_x):
                self.values['x'][i, 0:sequence_length[i]] = input_sequence[i]

            self.values['seq_length'] = np.array(sequence_length)
            self.values['y'] = np.array(output_characters)  # not really sure what to do with this... prob. embedding?

        else:
            self.add_data((input_sequence, sequence_length, output_characters))

