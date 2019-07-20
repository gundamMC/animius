import numpy as np
import librosa


def sliding_window(data, size, stepsize=1, axis=0):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    return strided


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

        return sliding_window((mfcc_data.transpose()), window, step)  # [number of conv frames x window x num_cepstral]
