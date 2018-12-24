import animius as am
from os.path import join
from os import mkdir, strerror
import errno
import json


class Waifu:

    def __init__(self, name, models, word_embedding):

        if 'CombinedPrediction' not in models:
            raise ValueError('A path to a combined prediction model (with key \'CombinedPrediction\') is required')

        self.combined_prediction = am.Chatbot.CombinedPredictionModel(models['CombinedPrediction'])

        self.input_data = am.ModelData.CombinedPredictionData(models['CombinedPrediction'].model_config)
        self.input_data.add_embedding_class(word_embedding)

        self.config = {'name': name, 'models': models}

    def predict(self, sentence):

        self.input_data.set_parse_input(sentence)

        return self.combined_prediction.predict(self.input_data)

    def save(self, directory):

        try:
            mkdir(directory)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                print('OS error: {0}'.format(exc))
                return
            pass

        with open(join(directory, 'waifu.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

    @classmethod
    def load(cls, directory):

        try:
            with open(join(directory, 'waifu.json'), 'r') as f:
                config = json.load(f)
                return cls(config['name'], config['models'])
        except OSError as exc:
            print('OS error: {0}'.format(exc))
        except KeyError:
            print('Load failed. Waifu.json is missing values (\'name\' and \'models\')')
