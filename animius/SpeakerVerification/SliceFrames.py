# IMPORTANT:

# This file is no longer used in Project Waifu.
# For now, it remains as a backup in case of future references

# edited from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
import collections
import contextlib
import os
import sys
import wave

import webrtcvad
from pydub import AudioSegment


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with wave.open(path, 'rb') as wf:
        num_channels = wf.getnchannels()

        # converts the wav to mono if it is stereo
        if num_channels == 2:
            wf.close()
            sound = AudioSegment.from_wav(path)
            sound = sound.set_channels(1)
            sound.export(path, format='wav')
            wf = wave.open(path, 'rb')

        sample_width = wf.getsamamidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 48000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        wf.close()
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsamamidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        sys.stdout.write(
            '1' if vad.is_speech(frame.bytes, sample_rate) else '0')
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def main(args):
    currentDir = os.path.dirname(os.path.realpath(__file__))

    if len(args) < 4:
        sys.stderr.write(
            'Usage: example.py <aggressiveness> <frame length> <padding length> <path to wav file>\n')
        sys.exit(1)

    waifuGUI = False
    if (len(args) > 4 and args[4] == "WaifuGUI"):
        count = 0
        waifuGUI = True

    for wavPath in args[3].split(","):

        folderName = os.path.basename(wavPath).replace(".wav", "")
        if (".WAV" in folderName):
            folderName = folderName.replace(".WAV", "")  # for the ppl that use all caps

        folderPath = currentDir + '/chunks/' + folderName

        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        audio, sample_rate = read_wave(wavPath)
        vad = webrtcvad.Vad(int(args[0]))
        frames = frame_generator(int(args[1]), audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, int(args[1]), int(args[2]), vad, frames)
        for i, segment in enumerate(segments):
            path = folderPath + '/chunk-%003d.wav' % (i,)
            write_wave(path, segment, sample_rate)
        print(wavPath + " Done!")

        # waifuGUI output
        if (waifuGUI):
            count += 1
            print("WaifuGUI: " + str(count))


if __name__ == '__main__':
    main(sys.argv[1:])
