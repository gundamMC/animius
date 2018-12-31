from os import mkdir
from os.path import join
import errno
import json
import animius as am


class ModelConfig:
    def __init__(self, cls=None, config=None, hyperparameters=None, model_structure=None):
        if config is None:
            self.config = {}
        else:
            self.config = config

        if hyperparameters is None:
            self.hyperparameters = {}
        else:
            self.hyperparameters = hyperparameters

        if model_structure is None:
            self.model_structure = {}
        else:
            self.model_structure = model_structure

        if cls is not None:
            self.apply_defaults(cls)

        # storing configs for saving/loading
        self.saved_directory = None
        self.saved_name = None

    def apply_defaults(self, cls):

        # cls may be a string class name
        if isinstance(cls, str):
            if cls == 'IntentNERModel':
                cls = am.IntentNER.IntentNERModel
            elif cls == 'ChatbotModel' or cls == 'CombinedChatbotModel':
                cls = am.Chatbot.ChatbotModel  # they share the same default model config
            elif cls == 'SpeakerVerificationModel':
                cls = am.SpeakerVerification.SpeakerVerificationModel
            else:
                raise ValueError('class not found with name {0}'.format(cls))

        for key, default_value in cls.DEFAULT_CONFIG().items():
            if key not in self.config:
                self.config[key] = default_value

        for key, default_value in cls.DEFAULT_HYPERPARAMETERS().items():
            if key not in self.hyperparameters:
                self.hyperparameters[key] = default_value

        for key, default_value in cls.DEFAULT_MODEL_STRUCTURE().items():
            if key not in self.model_structure:
                self.model_structure[key] = default_value

    def save(self, directory=None, name='model_config'):
        """
        Save a model config object to a directory

        :param directory: directory to save the config
        :param name: string to name the saved files
        :return: the directory in which the config is saved
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

        with open(join(directory, name + '.json'), 'w') as f:
            json.dump({
                'config': self.config,
                'hyperparameters': self.hyperparameters,
                'model_structure': self.model_structure
            }, f, indent=4)

        self.saved_directory = directory
        self.saved_name = name

        return directory

    @classmethod
    def load(cls, directory, name='model_config'):
        """
        Load a model config object from a saved directory

        :param directory: path to the directory in which the model config is saved
        :param name: name of the saved files
        :return: a model config object
        """

        with open(join(directory, name + '.json'), 'r') as f:
            stored = json.load(f)

        model_config = cls(config=stored['config'],
                           hyperparameters=stored['hyperparameters'],
                           model_structure=stored['model_structure'])

        model_config.saved_directory = directory
        model_config.saved_name = name

        return model_config
