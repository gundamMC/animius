import errno
import json
from os import mkdir
from os.path import join

import animius as am


class Waifu:

    def __init__(self, name, models=None):

        self.combined_prediction = None
        self.input_data = None

        if models is None:
            models = {}

        self.config = {'name': name, 'models': models}

    def add_combined_prediction_model(self, directory):

        if self.combined_prediction is not None:
            self.combined_prediction.close()
            print('Waifu {0}: Closing existing combined prediction model'.format(self.config['name']))

        self.combined_prediction = am.Chatbot.CombinedPredictionModel(directory)

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

    def save(self, directory):

        try:
            mkdir(directory)
            with open(join(directory, 'waifu.json'), 'w') as f:
                json.dump(self.config, f, indent=4)

        except OSError as exc:
            if exc.errno != errno.EEXIST:
                print('OS error: {0}'.format(exc))
                return
            pass

    @classmethod
    def load(cls, directory):
        try:
            with open(join(directory, 'waifu.json'), 'r') as f:
                config = json.load(f)

                waifu = cls(config['name'], config['models'])

                if 'CombinedPrediction' in config['models']:
                    waifu.load_combined_prediction_model()

                return waifu

        except OSError as exc:
            print('OS error: {0}'.format(exc))
        except KeyError:
            print('Load failed. Waifu.json is missing values (\'name\' and \'models\')')
