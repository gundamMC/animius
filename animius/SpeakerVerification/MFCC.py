import numpy as np
import scipy.io.wavfile as wav
import speechpy


class MFCC:

    @staticmethod
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

    @staticmethod
    def get_MFCC(path, window=10, step=3, num_cepstral=39, flatten=False):
        # read wav
        sample_rate, signal = wav.read(path)

        if signal.ndim == 2 and signal.shape[1] == 2:
            signal = (signal[:, 0] + signal[:, 1]) / 2

        mfcc_data = speechpy.feature.mfcc(signal, sampling_frequency=sample_rate, frame_length=0.020, frame_stride=0.01,
                                          num_cepstral=num_cepstral, num_filters=40, fft_length=512, low_frequency=0)
        # returns a matrix [number_of_frames x 39]

        mfcc_data = np.float32(mfcc_data)

        data = []

        if flatten:
            for i in range(window, mfcc_data.shape[0], step):
                data.append(mfcc_data[i - window:i].flatten('C'))
        else:
            return MFCC.sliding_window(mfcc_data, window, step)

        return np.array(data, dtype='float32')  # (number_of_frames - 10) x 390
