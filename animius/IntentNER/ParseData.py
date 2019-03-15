import json
import os

import numpy as np

import animius as am


class Parse:
    entity_to_index = None
    entities = None
    intent_to_index = None
    intents = None

    @staticmethod
    def get_ner_data(json_text):

        input_data = []  # words
        output_ner = []  # indexes of the classes of each word

        for text in json_text:
            input_data.extend(str.split(str.lower(text["text"])))
            if "entity" in text:
                output_ner.extend([Parse.entity_to_index[text["entity"]]] * len(str.split(text["text"])))
            else:
                output_ner.extend([0] * len(str.split(text["text"])))

        return input_data, output_ner

    @staticmethod
    def get_file_data(intent, words_to_index, data_folder, max_seq=20):
        data = json.load(open(os.path.join(data_folder, intent + ".json"), encoding="utf8"))
        data = data[intent]
        result_in = []
        result_length = []
        ner_out = []

        for i in data:
            input_data, output_ner = Parse.get_ner_data(i["data"])
            output_ner = am.Utils.set_sequence_length(output_ner, 0, max_seq=max_seq)
            ner_out.append(np.eye(8)[output_ner])
            input_data, input_length, _ = am.Utils.sentence_to_index(input_data, words_to_index, max_seq=max_seq,
                                                                     go=False, eos=False)
            result_in.append(input_data)
            result_length.append(input_length)

        intent_out = np.zeros((len(result_in), len(Parse.intent_to_index)))
        intent_out[:, Parse.intent_to_index[intent]] = 1

        return result_in, result_length, intent_out, ner_out

    @staticmethod
    def get_data(data_folder, word_embedding, max_seq=20):

        if not isinstance(word_embedding, am.WordEmbedding):
            raise TypeError('word embedding must be WordEmbedding object')

        Parse.get_labels(data_folder)

        if Parse.entity_to_index is None or Parse.intent_to_index is None:
            raise ValueError("Intent and NER labels not found. Please include the labels in the labels.json file")

        x = []
        x_length = []
        y_intent = []
        y_ner = []
        for filename in os.listdir(data_folder):
            filename = filename.split('.')[0]
            if filename in Parse.intent_to_index:
                result_in, result_length, intent_out, ner_out = Parse.get_file_data(filename,
                                                                                    word_embedding.words_to_index,
                                                                                    data_folder, max_seq)
                x.append(result_in)
                x_length.append(result_length)
                y_intent.append(intent_out)
                y_ner.append(ner_out)

        return np.vstack([i for i in x]), np.hstack([i for i in x_length]), np.vstack([i for i in y_intent]), np.vstack(
            [i for i in y_ner])

    @staticmethod
    def get_labels(data_folder):

        if 'labels.json' in os.listdir(data_folder):
            labels = json.load(open(os.path.join(data_folder, "labels.json"), encoding="utf8"))
            Parse.entities = labels['entities']
            Parse.intents = labels['intents']
            Parse.entity_to_index = dict(zip(Parse.entities, range(len(Parse.entities) - 1))).pop('none')
            Parse.intent_to_index = dict(zip(Parse.intents, range(len(Parse.intents) - 1)))
