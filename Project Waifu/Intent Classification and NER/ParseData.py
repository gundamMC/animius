import json

from pprint import pprint

entity_to_index = dict(object_name=1, object_type=2, time=3, location_name=4, condition=5, question=6, number=7)


data = json.load(open(".\\data\\SearchCreativeWork.json"))


data = data["SearchCreativeWork"]
# data is now a list of dictionaries with the name "data"

print(len(data))


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


for i in data:
    print(get_ner_data(i["data"]))



