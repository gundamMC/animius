import json
from numpy import eye, zeros, vstack
import os
from ProjectWaifu.Utils import sentence_to_index
import ProjectWaifu.WordEmbedding as Embedding

entity_to_index = dict(person_name=1, object_name=2, object_type=3, time=4, location_name=5, condition=6, info=7)
intent_to_index = dict(Chat=0, Positive=1, Negative=2, GetCreativeWork=3, GetPlace=4, GetWeather=5, PlayMusic=6,
                       GetTime=7, GetHardware=8, OpenExplorer=9, SetReminder=10, GetReminders=11, SetTimer=12,
                       SearchOnline=13, SetNote=14)


def get_ner_data(json_text):

    input_data = []   # words
    output_data = []  # indexes of the classes of each word

    for text in json_text:
        input_data.extend(str.split(str.lower(text["text"])))
        if "entity" in text:
            output_data.extend([entity_to_index[text["entity"]]] * len(str.split(text["text"])))
        else:
            output_data.extend([0] * len(str.split(text["text"])))

    return input_data, output_data


def get_file_data(intent, words_to_index, data_folder, max_seq=20):
    data = json.load(open(data_folder + "\\" + intent + ".json"))
    data = data[intent]
    result_in = []
    result_length = []
    ner_out = []

    for i in data:
        input_data, output_data = get_ner_data(i["data"])
        input_data, input_length = sentence_to_index(words_to_index, input_data, max_seq)
        result_in.append(input_data)
        result_length.append(input_length)
        ner_out.append(eye(8)[output_data])

    intent_out = zeros((len(result_in), len(intent_to_index)))
    intent_out[:, intent_to_index[intent]] = 1

    return result_in, result_length, intent_out, ner_out


def get_data(data_folder, word_embedding, max_seq=20):

    if not isinstance(word_embedding, Embedding.WordEmbedding):
        raise TypeError('word embedding must be WordEmbedding object')

    x = []
    x_length = []
    y_intent = []
    y_ner = []
    for filename in os.listdir(data_folder):
        filename = filename.split('.')[0]
        if filename in intent_to_index:
            result_in, result_length, intent_out, ner_out = get_file_data(filename, word_embedding.words_to_index, data_folder, max_seq)
            x.append(result_in)
            x_length.append(result_length)
            y_intent.append(intent_out)
            y_ner.append(ner_out)

    return vstack([i for i in x]), vstack([i for i in x_length]), vstack([i for i in y_intent]), vstack([i for i in y_ner])
