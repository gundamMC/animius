from abc import ABC, abstractmethod
import tensorflow as tf


class Model(ABC):

    def __init__(self, model_config):
        self.config = model_config
        self.model_structure = model_config.model_structure
        self.hyperparameters = model_config.hyperparameters
        self.data = None

        # Tensorflow initialization
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    @abstractmethod
    def train(self, epochs):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def predict_all(self, path, savePath=None):
        pass

    @abstractmethod
    def restore(self, path):
        pass

    def load_data(self, data):
        self.data = data
