from abc import ABC, abstractmethod
import tensorflow as tf
import Animius.ModelClasses as ModelClasses
import json
from os.path import isdir, join


class Model(ABC):

    @staticmethod
    def DEFAULT_CONFIG():
        return {
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
        return ModelClasses.ModelConfig(config=cls.DEFAULT_CONFIG(),
                                        model_structure=cls.DEFAULT_MODEL_STRUCTURE(),
                                        hyperparameters=cls.DEFAULT_HYPERPARAMETERS())

    def __init__(self, model_config, data, restore_path=None):

        if restore_path is not None:
            self.restore_config(restore_path)
            self.data = data
            return

        if not isinstance(model_config, ModelClasses.ModelConfig):
            raise TypeError('model_config must be a ModelConfig object')

        # apply values
        self.config = model_config.config
        self.model_structure = model_config.model_structure
        self.hyperparameters = model_config.hyperparameters
        self.data = data

        # prep for tensorflow
        self.saver = None
        self.tensorboard_writer = None
        self.sess = None

        # prep for hyperdash
        self.hyperdash = None

    def init_tensorflow(self):
        # Tensorflow initialization
        self.saver = tf.train.Saver()
        if self.config['tensorboard'] is not None:
            self.tensorboard_writer = tf.summary.FileWriter(self.config['tensorboard'])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())

    def init_hyerdash(self, name):
        if name is not None:
            from hyperdash import Experiment
            self.hyperdash = Experiment(name)

    def init_restore(self, restore_path, word_embedding_placeholder=None):
        # restore model data values
        if restore_path is not None:
            self.restore_model(restore_path)
            return  # do not restore word embedding

        # Do not include word embedding when restoring models
        if word_embedding_placeholder is not None and 'embedding' in self.data.values:
            embedding_placeholder = tf.placeholder(tf.float32, shape=self.data['embedding'].embedding.shape)
            self.sess.run(word_embedding_placeholder.assign(embedding_placeholder),
                          feed_dict={embedding_placeholder: self.data['embedding'].embedding})

    @abstractmethod
    def train(self, epochs):
        pass

    @abstractmethod
    def predict(self, input_data, save_path=None):
        pass

    def restore_config(self, path='./model'):

        if not isdir(path):
            raise NotADirectoryError('Save path must be a directory')

        with open(join(path, 'model_config.modelconfig'), 'r') as f:
            stored = json.load(f)
            self.config = stored['config']
            self.model_structure = stored['model_structure']
            self.hyperparameters = stored['hyperparameters']

    def restore_model(self, path='./model'):
        if not isdir(path):
            raise NotADirectoryError('Save path must be a directory')

        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))

    def set_data(self, data):
        self.data = data

    def save(self, path='./model/', meta=False):

        if not isdir(path):
            raise NotADirectoryError('Save path must be a directory')

        self.saver.save(self.sess, join(path + 'model'), global_step=self.config['epoch'], write_meta_graph=meta)
        with open(join(path, 'model_config.modelconfig'), 'w') as f:
            f.write(
                json.dumps(
                    {
                        'config': self.config,
                        'model_structure': self.model_structure,
                        'hyperparameters': self.hyperparameters
                    }, indent=4)
            )

    def close(self):
        self.sess.close()
