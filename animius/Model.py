import errno
import json
from abc import ABC, abstractmethod
from os import mkdir
from os.path import join

import tensorflow as tf

import animius as am


class Model(ABC):

    @staticmethod
    def DEFAULT_CONFIG():
        return {
            'name': 'Model',
            'epoch': 0,
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
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config, graph=graph)

            if init_param:
                self.sess.run(tf.global_variables_initializer())

    def init_hyerdash(self, name):
        if name is not None:
            from hyperdash import Experiment
            self.hyperdash = Experiment(name)

    def restore_embedding(self, word_embedding_placeholder):
        # Do not include word embedding when restoring models
        if word_embedding_placeholder is not None and 'embedding' in self.data.values:
            with self.sess.graph.as_default():
                embedding_placeholder = tf.placeholder(tf.float32, shape=self.data['embedding'].embedding.shape)
                self.sess.run(word_embedding_placeholder.assign(embedding_placeholder),
                              feed_dict={embedding_placeholder: self.data['embedding'].embedding})

    @abstractmethod
    def train(self, epochs):
        pass

    @abstractmethod
    def predict(self, input_data, save_path=None):
        pass

    def restore_config(self, directory='./model'):
        try:
            with open(join(directory, 'model_config.json'), 'r') as f:
                stored = json.load(f)
                self.config = stored['config']
                self.model_structure = stored['model_structure']
                self.hyperparameters = stored['hyperparameters']
        except OSError as exc:
            print('OS error: {0}'.format(exc))
        except KeyError:
            print('Restore failed. model_config.json is missing values')

    def restore_model(self, directory='./model'):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))

    def set_data(self, data):
        self.data = data

    def save(self, directory='./model/', meta=True, graph=False):

        try:
            mkdir(directory)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                print('OS error: {0}'.format(exc))
                return
            pass

        self.saver.save(self.sess, join(directory, 'model'), global_step=self.config['epoch'], write_meta_graph=meta)

        if graph:
            tf.train.write_graph(self.sess.graph.as_graph_def(), directory, 'model_graph.pb', as_text=False)
            self.config['graph'] = join(directory, 'model_graph.pb')

        with open(join(directory, 'model_config.json'), 'w') as f:
            json.dump({
                'config': self.config,
                'model_structure': self.model_structure,
                'hyperparameters': self.hyperparameters
            }, f, indent=4)

        print('Model saved at ' + directory)

    @classmethod
    def load(cls, directory, data=None):
        pass

    def close(self):
        self.sess.close()
