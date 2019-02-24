import errno
import json
from os import mkdir
from os.path import join

import animius as am


class Waifu:

    def __init__(self, name, models=None, description=''):

        self.combined_prediction = None
        self.input_data = None

        if models is None:
            models = {}

        self.config = {'name': name, 'description': description, 'data': None, 'models': models}

        self.saved_directory = None
        self.saved_name = None

    def add_combined_prediction_model(self, directory, name):

        if self.combined_prediction is not None:
            self.combined_prediction.close()
            print('Waifu {0}: Closing existing combined prediction model'.format(self.config['name']))

        self.combined_prediction = am.Chatbot.CombinedPredictionModel(directory, name)

        if 'CombinedPrediction' in self.config['models']:
            print('Waifu {0}: Overwriting existing combined prediction model'.format(self.config['name']))

        self.config['models']['CombinedPrediction'] = directory

    def load_combined_prediction_model(self):

        if self.combined_prediction is not None:
            self.combined_prediction.close()
            print('Waifu {0}: Closing existing combined prediction model'.format(self.config['name']))

        if 'CombinedPrediction' not in self.config['models']:
            print('Waifu {0}: No combined prediction model found.'.format(self.config['name']))

        self.combined_prediction = am.Chatbot.CombinedPredictionModel(self.config['models']['CombinedPrediction'])

    def build_input(self, embedding):
        self.input_data = am.ModelData.CombinedPredictionData(self.combined_prediction.model_config)
        self.input_data.add_embedding_class(embedding)

    def predict(self, sentence):

        self.input_data.set_parse_input(sentence)

        return self.combined_prediction.predict(self.input_data)

    def save(self, directory, name='waifu'):

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

        # save input data
        self.input_data.save(directory=directory, name=name + '_input_data', save_embedding=True)
        self.config['data'] = name + '_input_data'

        with open(join(directory, name + '.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        self.saved_directory = directory
        self.saved_name = name

        return directory

    @classmethod
    def load(cls, directory, name='waifu'):
        with open(join(directory, name + '.json'), 'r') as f:
            config = json.load(f)

        waifu = cls(config['name'], config['models'], config['description'])

        # load models
        if 'CombinedPrediction' in config['models']:
            waifu.load_combined_prediction_model()

        # set up input data
        if 'data' in config and 'data' is not None:
            waifu.input_data = am.CombinedPredictionData.load(directory, name + '_input_data')

        waifu.saved_directory = directory
        waifu.saved_name = name

        return waifu
