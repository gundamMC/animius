import os

import pysubs2
from pydub import AudioSegment


class Parser:

    def __init__(self):
        self.SSAFile = None
        self.audio_sentences = None

    def load(self, path):
        self.SSAFile = pysubs2.SSAFile.load(path)

    def slice_audio(self, path):
        print("Processing " + path)

        audio = AudioSegment.from_file(path)
        savePath = ".\\audio\\" + os.path.splitext(os.path.basename(path))[0] + "\\"
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        index = 0
        for sub in self.SSAFile:
            if sub.duration < 200:  # skip ones that are shorter than 0.2 seconds
                continue
            segment = audio[sub.start:sub.end]
            segment.export(savePath + str(index).zfill(4) + ".wav", format="wav")
            index += 1
            self.audio_sentences.append(sub.plaintext)

        print("Done!")

    def detect_conversation(self, speaking):
        window = 5
        conversation = []

        if isinstance(speaking, str):
            with open(speaking, 'r') as file:
                lines = file.read().splitlines()
                speaking = []
                for line in lines:
                    speaking.append(line.split()[0] == "True")

        print(speaking)

        # get conservation frames by comparing the is-speaker and not-speaker labels within a window
        for i in range(0, len(speaking) - window, window):
            TrueCount = 0
            FalseCount = 0
            for j in range(window):
                if speaking[i + j]:
                    TrueCount += 1
                else:
                    FalseCount += 1
            if TrueCount > 0 and FalseCount > 0:
                conversation.append([i, i + window])

        result = []
        startingIndex = 0
        # make connecting conversation frames as one large group
        while startingIndex < len(conversation) - 1:
            for i in range(startingIndex + 1, len(conversation)):
                if conversation[i][0] != conversation[i - 1][1]:
                    # if the ending does not follow the previous start, the conversation is not connected
                    result.append([conversation[startingIndex][0], conversation[i - 1][1]])
                    break
                elif i == len(conversation) - 1:
                    # if it has reached the end of conversations
                    result.append([conversation[startingIndex][0], conversation[i - 1][1]])
                    break
                else:
                    continue
            startingIndex = i  # start from where the for loop left off

        return result  # a list of [start, end]'s

    def get_conversation_sentences(self, conversations):
        # return the actual sentences in conversations
        sentences = []
        for conv in conversations:
            conv_sentences = []
            for i in range(conv[0], conv[1] + 1):
                conv_sentences.append(self.audio_sentences[i])
            sentences.append(conv_sentences)
        return sentences

    def get_chatbot_data(self, is_speaker):
        conversations = self.detect_conversation(is_speaker)
        queries = []
        responses = []

        for conv in conversations:

            i = conv[0]
            start_index = None  # the index where the first non-speaker speaks

            while i <= conv[1]:
                if not is_speaker[i]:
                    start_index = i
                    break

            if start_index is None:
                # the 'conversation' for some reason is a monologue of the speaker
                continue

            i = start_index
            last_is_speaker = False
            not_speaker_query = self.audio_sentences[i]
            speaker_response = ''

            while i <= conv[1]:
                if is_speaker[i]:
                    speaker_response += self.audio_sentences[i]
                    last_is_speaker = True
                else:
                    # switch from speaker to non-speaker = the end of a query and response set
                    if last_is_speaker is True:
                        queries.append(not_speaker_query)
                        responses.append(speaker_response)
                        not_speaker_query = ''
                        speaker_response = ''

                    not_speaker_query += self.audio_sentences[i]
                    last_is_speaker = False

        return queries, responses  # add to chatbot data with add_parse_sentences
