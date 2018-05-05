from abc import ABC, abstractmethod
import tensorflow as tf


class Network(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    # returns the result of the network without sigmoid
    @abstractmethod
    def network(self):
        pass
    
    @abstractmethod
    def setTrainingData(self):
        pass

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
