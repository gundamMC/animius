from abc import ABC, abstractmethod
import tensorflow as tf


class Model(ABC):

    def __init__(self, model_config):
        self.config = model_config.config
        # for easy access
        self.model_structure = model_config.config['model_structure']
        self.hyperparameters = model_config.config['hyperparameters']
        self.data = None

        # Tensorflow initialization
        self.saver = tf.train.Saver()
        if self.config['tensorboard'] is not None:
            self.tensorboard_writer = tf.summary.FileWriter(self.config['tensorboard'])
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.epoch = 0

    @abstractmethod
    def train(self, epochs):
        pass

    @abstractmethod
    def predict(self, input_data, save_path=None):
        pass

    @abstractmethod
    def restore(self, path):
        pass

    def load_data(self, data):
        self.data = data

    def save(self, path='./model/model', meta=False):
        self.saver.save(self.sess, path, global_step=self.epoch, write_meta_graph=meta)
