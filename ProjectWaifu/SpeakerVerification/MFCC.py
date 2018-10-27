import numpy as np
import speechpy
import scipy.io.wavfile as wav


def get_MFCC(path, window=10, step=3, num_cepstral=39, flatten=False):
    # read wav
    sample_rate, signal = wav.read(path)

    if signal.ndim == 2 and signal.shape[1] == 2:
        signal = (signal[:, 0] + signal[:, 1]) / 2

    mfcc_data = speechpy.feature.mfcc(signal, sampling_frequency=sample_rate, frame_length=0.020, frame_stride=0.01,
                                      num_cepstral=num_cepstral, num_filters=40, fft_length=512, low_frequency=0)
    # returns a matrix [number_of_frames x 39]

    assert isinstance(mfcc_data, np.ndarray)

    data = []

    if flatten:
        for i in range(window, mfcc_data.shape[0], step):
            data.append(mfcc_data[i - window:i].flatten('C'))
    else:
        for i in range(window, mfcc_data.shape[0], step):
            data.append(mfcc_data[i - window:i])

    return np.array(data)  # (number_of_frames - 10) x 390
