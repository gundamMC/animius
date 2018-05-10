import json
from numpy import eye, zeros, vstack
import os
import ProjectWaifu.Utils as Utils

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


def get_intent_data(intent, word_vec, data_folder):
    data = json.load(open(data_folder + intent + ".json"))
    data = data[intent]
    result_in = []
    ner_out = []
    for i in data:
        input_data, output_data = get_ner_data(i["data"])
        input_data = sentence_to_vec(word_vec, input_data)
        output_data.extend([0] * (30 - len(output_data)))
        result_in.append(input_data)
        ner_out.append(eye(8)[output_data])

    return result_in, ner_out


def get_data(word_vec, data_folder):
    input = []
    output_ner = []
    output_intent = []
    for filename in os.listdir(data_folder):
        filename = filename.split('.')[0]
        if filename in intent_to_index:
            result_in, ner_out = get_intent_data(filename, word_vec, data_folder)
            intent_out = zeros((len(ner_out), len(intent_to_index)))
            intent_out[:, intent_to_index[filename]] = 1

            input.append(result_in)
            output_ner.append(ner_out)
            output_intent.append(intent_out)

    if len(input) == 0:
        Utils.printMessage("Error: No intent found")
        exit()

    return vstack([i for i in input]), vstack([i for i in output_ner]), vstack([i for i in output_intent])


def sentence_to_vec(word_vec, sentence):

    result = []

    for word in sentence:
        if word in word_vec:
            result.append(word_vec[word])
        else:
            result.append(word_vec["<unknown>"])

    for i in range(30 - len(result)):
        result.append([0] * 50)

    return result
