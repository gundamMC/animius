import numpy as np
import speechpy
import scipy.io.wavfile as wav

def getData(path, flatten = True):

    # read wav
    sample_rate, signal = wav.read(path)

    if signal.ndim == 2 and signal.shape[1] == 2:
        signal = (signal[:,0] + signal[:,1]) / 2

    mfcc_data = speechpy.feature.mfcc(signal, sampling_frequency=sample_rate, frame_length=0.020, frame_stride=0.01, num_cepstral = 39 ,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    # returns a matrix [number_of_frames x 39]

    data = []

    if(flatten):
        for i in range(10, mfcc_data.shape[0], 3):
            data.append(mfcc_data[i - 10 : i].flatten('C'))
    else:
        for i in range(10, mfcc_data.shape[0], 3):
            data.append(mfcc_data[i - 10 : i])

    print(path, "processed")

    return np.array(data) # (number_of_frames - 10) x 390