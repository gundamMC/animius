import pysubs2
from pydub import AudioSegment
import os


class Parser:

    def __init__(self):
        self.SSAFile = None

    def load(self, path):
        self.SSAFile = pysubs2.SSAFile.load(path)

    def sliceAudio(self, path):
        print("Processing " + path)

        audio = AudioSegment.from_file(path)
        savePath = ".\\audio\\" + os.path.splitext(os.path.basename(path))[0] + "\\"
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        index = 0
        for sub in self.SSAFile:
            if "\\" in sub.text:  # no weird characters
                continue
            segment = audio[sub.start:sub.end]
            segment.export(savePath + str(index).zfill(4) + "-" + sub.text + ".wav", format="wav")
            index += 1

        print("Done!")

    def detectConversation(self, speaking):
        window = 5
        conversation = []

        if isinstance(speaking, str):
            with open(speaking) as file:
                lines = file.read().splitlines()
                speaking = []
                for line in lines:
                    speaking.append(line.split()[0] == "True")

        print(speaking)

        for i in range(0, len(speaking) - window, window):
            TrueCount = 0
            FalseCount = 0
            for j in range(window):
                if speaking[i+j]:
                    TrueCount += 1
                else:
                    FalseCount += 1
            conversation.extend([TrueCount > 0 and FalseCount > 0] * window)
        return conversation
