import errno
import json
from abc import ABC, abstractmethod
from os import mkdir
from os.path import join

import tensorflow as tf

import animius as am


class Model(ABC):

    @staticmethod
    def _get_default_device():
        if tf.test.is_gpu_available:
            return tf.test.gpu_device_name()
        else:
            return '/cpu:0'

    @staticmethod
    def DEFAULT_CONFIG():
        return {
            'device': Model._get_default_device(),
            'class': '',
            'epoch': 0,
            'cost': None,
            'display_step': 1,
            'tensorboard': None,
            'hyperdash': None
        }

    @staticmethod
    def DEFAULT_MODEL_STRUCTURE():
        return {}

    @staticmethod
    def DEFAULT_HYPERPARAMETERS():
        return {}

    @classmethod
    def DEFAULT_MODEL_CONFIG(cls):
        return am.ModelConfig(cls,
                              config=cls.DEFAULT_CONFIG(),
                              model_structure=cls.DEFAULT_MODEL_STRUCTURE(),
                              hyperparameters=cls.DEFAULT_HYPERPARAMETERS())

    def __init__(self):
        # Attributes assigned at init function (manually called by the user)
        self.config = None
        self.model_structure = None
        self.hyperparameters = None

        self.data = None

        # prep for tensorflow
        self.graph = None
        self.saver = None
        self.tensorboard_writer = None
        self.sess = None

        # prep for hyperdash
        self.hyperdash = None

        # save/load
        self.saved_directory = None
        self.saved_name = None

    @abstractmethod
    def build_graph(self, model_config, data):
        pass

    # Tensorflow initialization
    def init_tensorflow(self, graph=None, init_param=True, init_sess=True):

        if graph is not None:
            self.graph = graph

        with self.graph.as_default():
            if self.saver is None:
                self.saver = tf.train.Saver()
            if self.config['tensorboard'] is not None:
                self.tensorboard_writer = tf.summary.FileWriter(self.config['tensorboard'])

            if init_sess:

                # force cpu utilization
                if self.config['device'] == '/cpu:0':
                    config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0}, allow_soft_placement=True)
                else:  # gpu allow growth
                    config = tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config, graph=graph)

            if init_param:
                self.sess.run(tf.global_variables_initializer())

    def init_hyperdash(self, name):
        if name is not None:
            from hyperdash import Experiment
            self.hyperdash = Experiment(name)

    def init_embedding(self, word_embedding_placeholder):
        # Do not include word embedding when restoring models
        if word_embedding_placeholder is not None and 'embedding' in self.data.values:
            with self.sess.graph.as_default():
                embedding_placeholder = tf.placeholder(tf.float32, shape=self.data['embedding'].embedding.shape)
                self.sess.run(word_embedding_placeholder.assign(embedding_placeholder),
                              feed_dict={embedding_placeholder: self.data['embedding'].embedding})
        else:
            raise ValueError('Embedding not found.')

    @abstractmethod
    def train(self, epochs, CancellationToken):
        pass

    @abstractmethod
    def predict(self, input_data, save_path=None):
        pass

    def restore_config(self, directory, name='model'):
        with open(join(directory, name + '.json'), 'r') as f:
            stored = json.load(f)
            self.config = stored['config']
            self.model_structure = stored['model_structure']
            self.hyperparameters = stored['hyperparameters']

    def restore_model(self, directory):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))

    def set_data(self, data):
        self.data = data

    def save(self, directory=None, name='model', meta=True, graph=False):

        # model config and graph must be initiated before saving
        if self.config is None:
            raise ValueError("Model config and graph must be initiated before saving")

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

        self.saver.save(self.sess, join(directory, name), global_step=self.config['epoch'], write_meta_graph=meta)
        if graph:
            tf.train.write_graph(self.sess.graph.as_graph_def(), directory, name + '_graph.pb', as_text=False)
            self.config['graph'] = join(directory, name + '_graph.pb')

        # saving an individual copy because config has been changed
        with open(join(directory, name + '.json'), 'w') as f:
            json.dump({
                'config': self.config,
                'model_structure': self.model_structure,
                'hyperparameters': self.hyperparameters
            }, f, indent=4)

        self.saved_directory = directory
        self.saved_name = name

        return directory

    @classmethod
    def load(cls, directory, name='model', data=None):
        with open(join(directory, name + '.json'), 'r') as f:
            stored = json.load(f)
            class_name = stored['config']['class']

        if class_name == 'Chatbot':
            return am.Chatbot.ChatbotModel.load(directory, name=name, data=data)
        elif class_name == 'CombinedChatbot':
            return am.Chatbot.CombinedChatbotModel.load(directory, name=name, data=data)
        elif class_name == 'IntentNER':
            return am.IntentNER.IntentNERModel.load(directory, name=name, data=data)
        elif class_name == 'SpeakerVerification':
            return am.SpeakerVerification.SpeakerVerificationModel.load(directory, name=name, data=data)
        else:
            raise ValueError("Loading failed: class name not found")

    def model_config(self):
        return am.ModelConfig(
            config=self.config,
            hyperparameters=self.hyperparameters,
            model_structure=self.model_structure)

    def close(self):
        self.sess.close()
