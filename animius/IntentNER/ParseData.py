import json
import os
import warnings


class Parse:
    entity_to_index = None
    entities = None
    intent_to_index = None
    intents = None

    @staticmethod
    def get_ner_data(json_text):
        # process ner data of a single sentence input

        input_data = []  # words
        output_ner = []  # indexes of the classes of each word

        for text in json_text:
            input_data.extend(str.split(str.lower(text["text"])))
            if "entity" in text:
                if text["entity"] not in Parse.entity_to_index:
                    warnings.warn("Entity label '{0}' does not exist. Replacing it with 0.".format(text['entity']))
                    Parse.entity_to_index[text['entity']] = 0
                output_ner.extend([Parse.entity_to_index[text["entity"]]] * len(str.split(text["text"])))
            else:
                output_ner.extend([0] * len(str.split(text["text"])))

        return input_data, output_ner

    @staticmethod
    def get_file_data(intent, data_folder):
        # process a single intent file

        data = json.load(open(os.path.join(data_folder, intent + ".json"), encoding="utf8"))
        data = data[intent]
        result_in = []
        ner_out = []

        for i in data:
            input_data, output_ner = Parse.get_ner_data(i["data"])
            result_in.append(input_data)
            ner_out.append(output_ner)

        intent_out = [Parse.intent_to_index[intent]] * len(ner_out)

        return result_in, intent_out, ner_out

    @staticmethod
    def get_data(data_folder):

        Parse.get_labels(data_folder)

        if Parse.entity_to_index is None or Parse.intent_to_index is None:
            raise ValueError("Intent and NER labels not found. Please include the labels in the labels.json file")

        x = []
        y_intent = []
        y_ner = []
        for filename in os.listdir(data_folder):
            filename = filename.split('.')[0]
            if filename in Parse.intent_to_index:  # only process listed files
                result_in, intent_out, ner_out = Parse.get_file_data(filename, data_folder)
                x.extend(result_in)
                y_intent.extend(intent_out)
                y_ner.extend(ner_out)

        return x, y_intent, y_ner

    @staticmethod
    def get_labels(data_folder):

        if 'labels.json' in os.listdir(data_folder):
            labels = json.load(open(os.path.join(data_folder, "labels.json"), encoding="utf8"))
            Parse.entities = labels['entities']
            Parse.intents = labels['intents']
            Parse.entity_to_index = dict(zip(Parse.entities, range(len(Parse.entities) - 1)))
            Parse.entity_to_index.pop('none')
            Parse.intent_to_index = dict(zip(Parse.intents, range(len(Parse.intents) - 1)))
