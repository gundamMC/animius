from abc import ABC, abstractmethod
import tensorflow as tf


class Network(ABC):

    @abstractmethod
    def train(self, epochs, display_step):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def predictAll(self, path, savePath=None):
        pass
    
    @abstractmethod
    def setTrainingData(self, data):
        pass

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
