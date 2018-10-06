from abc import ABC, abstractmethod
import tensorflow as tf
from ProjectWaifu.ModelClasses import ModelConfig
import json
from os.path import isdir, join


class Model(ABC):

    def DEFAULT_CONFIG(self):
        return {
            'epoch': 0,
            'display_step': 1,
            'tensorboard': './model/tensorboard',
            'hyperdash': False
        }

    @abstractmethod
    def DEFAULT_MODEL_STRUCTURE(self):
        return {}

    @abstractmethod
    def DEFAULT_HYPERPARAMETERS(self):
        return {}

    @staticmethod
    def apply_default(user_values, default_values):
        for key, default_value in default_values.items():
            if key not in user_values:
                user_values[key] = default_value

    def __init__(self, model_config, data, restore_path=None):

        if restore_path is not None:
            self.restore_config(restore_path)
            self.data = data
            return

        if not isinstance(model_config, ModelConfig):
            raise TypeError('model_config must be a ModelConfig object')

        self.config = model_config.config
        Model.apply_default(self.config, self.DEFAULT_CONFIG())
        self.model_structure = model_config.model_structure
        Model.apply_default(self.model_structure, self.DEFAULT_MODEL_STRUCTURE())
        self.hyperparameters = model_config.hyperparameters
        Model.apply_default(self.config, self.DEFAULT_HYPERPARAMETERS())
        self.data = data

        # prep for tensorflow
        self.saver = None
        self.tensorboard_writer = None
        self.sess = None

    def init_tensorflow(self):
        # Tensorflow initialization
        self.saver = tf.train.Saver()
        if self.config['tensorboard'] is not None:
            self.tensorboard_writer = tf.summary.FileWriter(self.config['tensorboard'])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    @abstractmethod
    def train(self, epochs):
        pass

    @abstractmethod
    def predict(self, input_data, save_path=None):
        pass

    def restore_config(self, path='./model'):

        if not isdir(path):
            raise NotADirectoryError('Save path must be a directory')

        with open(join(path, 'model_config.wmc'), 'r') as f:
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
        with open(join(path, 'model_config.wmc'), 'w') as f:
            json.dump(
                {
                    'config': self.config,
                    'model_structure': self.model_structure,
                    'hyperparameters': self.hyperparameters
                }, f)

    def close(self):
        self.sess.close()
