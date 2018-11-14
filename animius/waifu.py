import animius as am
import numpy as np


class Waifu:

    def __init__(self, models):
        self.intentNER = models['IntentNER']
        self.chatbot = models['Chatbot']

        assert isinstance(self.intentNER, am.IntentNER.IntentNERModel)
        assert isinstance(self.chatbot, am.Chatbot.ChatbotModel)

        self.intent_input_data = am.ModelClasses.IntentNERData(self.intentNER.config)
        self.chatbot_input_data = am.ModelClasses.ChatbotData(self.chatbot.config)

    def predict(self, sentence):

        self.intent_input_data.set_parse_input(sentence)

        intent, ner = self.intentNER.predict(self.intent_input_data)

        intent = am.IntentNER.Parse.intents[np.argmax(intent[0])]

        ner = [am.IntentNER.Parse.entities[i] for i in np.argmax(ner[0], axis=-1)]

        if intent == 0:  # intent is chat

            self.chatbot_input_data.set_parse_input(sentence)

            chatbot_response = self.chatbot.predict(self.chatbot_input_data)
            # returns a string (the sentence output)

            return intent, chatbot_response

        else:

            ner_tags = {}
            # split sentence into chunks
            sentence = am.Chatbot.Parse.split_sentence(sentence)

            for i in range(len(ner)):
                if ner[i] not in ner_tags:
                    ner_tags[ner[i]] = sentence[i]
                else:
                    ner_tags[ner[i]] += ' ' + sentence[i]

            return intent, ner
