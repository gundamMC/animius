import json
from numpy import eye

entity_to_index = dict(object_name=1, object_type=2, time=3, location_name=4, condition=5, question=6, number=7)


def get_ner_data(json_text):

    input_data = []   # words
    output_data = []  # indexes of the classes of each word

    for text in json_text:
        input_data.extend(str.split(text["text"]))
        if "entity" in text:
            output_data.extend([entity_to_index[text["entity"]]] * len(str.split(text["text"])))
        else:
            output_data.extend([0] * len(str.split(text["text"])))

    return input_data, output_data


def get_data(intent):
    data = json.load(open(".\\data\\" + intent + ".json"))
    data = data[intent]
    result_in = []
    result_out = []
    for i in data:
        input_data, output_data = get_ner_data(i["data"])
        input_data.extend(["<end>"] * (30 - len(input_data)))
        output_data.extend([0] * (30 - len(output_data)))
        result_in.append(input_data)
        result_out.append(eye(8)[output_data])

    return result_in, result_out
