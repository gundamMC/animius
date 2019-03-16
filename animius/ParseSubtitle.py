import pysubs2
from pydub import AudioSegment
from os import mkdir, path


class Parser:

    def __init__(self):
        self.SSAFile = None
        self.audio_sentences = []

    def load(self, subtitle_path):
        self.SSAFile = pysubs2.SSAFile.load(subtitle_path)

    def parse_audio_sentences(self):
        # only use this when not using slice_audio
        for sub in self.SSAFile:
            if sub.duration < 200 or sub.type == 'Comment':
                continue
            self.audio_sentences.append(sub.plaintext)

    def slice_audio(self, audio_path, save_path):
        audio = AudioSegment.from_file(audio_path)
        if not path.exists(save_path):
            mkdir(save_path)
        index = 0
        for sub in self.SSAFile:
            if sub.duration < 200 or sub.type == 'Comment':  # prevent short sentences and comments
                continue
            segment = audio[sub.start:sub.end]
            segment.export(path.join(save_path, str(index).zfill(4) + ".wav"), format="wav")
            index += 1
            self.audio_sentences.append(sub.plaintext)

    def detect_conversation(self, speaking, time_gap=5000):
        # time gap in milliseconds, default of 5 seconds
        time_barriers = [0]

        # search for long pauses between speech
        for i in range(1, len(self.audio_sentences)):
            if self.SSAFile[i].start - self.SSAFile[i-1].end > time_gap:
                time_barriers.append(i)
                # i is the start of a new time gap

        conversations = []

        for i in range(1, len(time_barriers)):

            start = time_barriers[i-1]
            end = time_barriers[i]

            x = ''
            y = ''

            for j in range(start, end):
                if speaking[j]:
                    # is speaker
                    if x == '':
                        # no input given
                        continue
                    else:
                        y += self.audio_sentences[j] + ' '

                        # ends with speaker answering
                        if j == end - 1:
                            conversations.append([x, y])
                else:
                    # is not speaker
                    if y == '':
                        # x has not yet been answered
                        x += self.audio_sentences[j] + ' '
                    else:
                        # y of previous x is answered
                        conversations.append([x, y])
                        x = self.audio_sentences[j]
                        y = ''

        return conversations
