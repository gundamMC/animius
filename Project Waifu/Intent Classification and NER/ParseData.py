import json

from pprint import pprint

data = json.load(open("Direction to data"))


data = data["SearchCreativeWork"]
# data is now a list of dictionaries with the name "data"

pprint(data[0])

for i in data:
    for text in i["data"]:
        print(text["text"], end='')
        if "entity" in text:
            print("(" + text["entity"] + ")", end='')
    print()  # new line
