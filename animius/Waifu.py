import errno
import json
import re
import shutil
from os import mkdir
from os.path import join, isfile, splitext

import animius as am


class Waifu:

    def __init__(self, name, models=None, description='', image=''):

        self.combined_chatbot = None

        if models is None:
            models = {}

        self.config = {'name': name, 'description': description, 'image': image,
                       'models': models, 'regex_rule': {}}

        self.saved_directory = None
        self.saved_name = None

        self.embedding = None

    def add_combined_chatbot_model(self, directory, name='model'):

        if self.combined_chatbot is not None:
            self.combined_chatbot.close()
            print('Waifu {0}: Closing existing combined chatbot model'.format(self.config['name']))

        self.combined_chatbot = am.Chatbot.CombinedChatbotModel.load(directory, name)

        if 'CombinedChatbot' in self.config['models']:
            print('Waifu {0}: Overwriting existing combined chatbot model'.format(self.config['name']))

        self.config['models']['CombinedChatbotDirectory'] = directory
        self.config['models']['CombinedChatbotName'] = name

    def load_combined_chatbot_model(self):

        if self.combined_chatbot is not None:
            self.combined_chatbot.close()
            print('Waifu {0}: Closing existing combined chatbot model'.format(self.config['name']))

        if 'CombinedChatbot' not in self.config['models']:
            print('Waifu {0}: No combined chatbot model found.'.format(self.config['name']))

        self.combined_chatbot = am.Chatbot.CombinedChatbotModel.load(
            self.config['models']['CombinedChatbotDirectory'], self.config['models']['CombinedChatbotName']
        )

    def add_embedding(self, embedding):
        self.embedding = embedding
        if self.combined_chatbot is not None:
            self.combined_chatbot.add_embedding(embedding)

    def add_regex(self, regex_rule, isIntentNER, result):
        self.config['regex_rule'][regex_rule] = [isIntentNER, result]

    def predict(self, sentence):

        regex_rule = self.config['regex_rule']

        # {"how's the weather in (.+)": [True, 'getWeather'], "good morning": [False, 'Good morning!']}

        for rule in regex_rule.keys():
            placeholder = re.findall(rule, sentence)

            if len(placeholder) != 0:
                if regex_rule[rule][0]:  # return intent and ner
                    count = 0

                    intent = regex_rule[rule]
                    ner = []
                    word_list = sentence.split(' ')
                    ner_sentence = []
                    for word in word_list:
                        if word in placeholder:
                            ner.append('placeholder_' + str(count))
                            count += 1
                        else:
                            ner.append('')

                        ner_sentence.append(word)

                    return {'intent': intent, 'ner': [ner, ner_sentence]}

                else:  # return chat
                    return {'message': regex_rule[rule][1]}

        result = self.combined_chatbot.predict(sentence)

        return result

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

        image = self.config['image']
        file_name = ''

        if isinstance(image, list):
            file_type = image[0]
            b64 = image[1]
            file_name = join(directory, name + '.' + file_type)
            with open(file_name, 'w') as f:
                f.write(b64.decode('base64'))

        elif isinstance(image, str):
            if isfile(image):
                file_name = join(directory, name + splitext(image)[1])
                shutil.copy(image, file_name)

        self.config['image'] = join(directory, file_name)

        # save embedding
        if self.embedding is not None:
            if self.embedding.saved_directory is None:  # embedding has not been saved
                self.embedding.save(join(directory, 'embedding'), 'embedding')

            self.config['embedding_directory'] = self.embedding.saved_directory
            self.config['embedding_name'] = self.embedding.saved_name

        # save config
        with open(join(directory, name + '.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        self.saved_directory = directory
        self.saved_name = name

        return directory

    @classmethod
    def load(cls, directory, name='waifu'):
        with open(join(directory, name + '.json'), 'r') as f:
            config = json.load(f)

        waifu = cls(config['name'], config['models'], config['description'], config['image'])

        # load models
        if 'CombinedChatbotDirectory' in config['models']:
            waifu.load_combined_chatbot_model()

        # load embedding
        if 'embedding_directory' in config:
            waifu.embedding = am.WordEmbedding.load(config['embedding_directory'], config['embedding_name'])
            waifu.add_embedding(waifu.embedding)

        waifu.saved_directory = directory
        waifu.saved_name = name

        return waifu
