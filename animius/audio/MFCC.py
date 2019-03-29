import numpy as np
# import scipy.io.wavfile as wav
# import speechpy
import librosa


def get_MFCC(path, sample_rate=None, num_cepstral=39, conv=False, window=20, step=5):
    # read wav
    # sample_rate, signal = wav.read(path)

    signal, sample_rate = librosa.load(path, sr=sample_rate)  # librosa will resample if sr is not None

    if signal.ndim == 2 and signal.shape[1] == 2:
        signal = (signal[:, 0] + signal[:, 1]) / 2

    # mfcc_data = speechpy.feature.mfcc(signal, sampling_frequency=sample_rate, frame_length=0.020, frame_stride=0.01,
    #                                   num_cepstral=num_cepstral, num_filters=40, fft_length=512, low_frequency=0)
    mfcc_data = librosa.feature.mfcc(signal, sr=sample_rate, n_mfcc=num_cepstral)
    # returns a matrix [39 x number_of_frames]

    assert isinstance(mfcc_data, np.ndarray)

    if not conv:
        return mfcc_data.transpose()  # [number of frames x num_cepstral]
    else:
        # conv with overlapping windows
        data = []
        mfcc_data = mfcc_data.transpose()
        for i in range(window, mfcc_data.shape[0], step):
            data.append(mfcc_data[i - window:i])

        return np.array(data)  # [number of conv frames x window x num_cepstral]
