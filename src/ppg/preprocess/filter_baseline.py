import numpy as np
from BaselineRemoval import BaselineRemoval
from scipy import signal


def bandpass_filter(waveform: np.ndarray, hp: float, lp: float, fs: float) -> np.ndarray:
    """
    4th order butterworth band-pass digital filter.

    Parameters:
        waveform (np.ndarray): Raw PPG signal waveform.
        hp (float): High-pass frequency.
        lp (float): Low-pass frequency.
        fs (float): Sample frequency of the signal (Hz).

    Returns:
        np.ndarray: Band-pass filtered PPG signal waveform.
    """

    order = 4
    b, a = signal.butter(order, (hp, lp), btype='bandpass', fs=fs)
    return signal.lfilter(b, a, waveform - waveform[0])


def lowpass_filter(waveform: np.ndarray, order: int, lp: float, fs: float) -> np.ndarray:
    """
    4th order butterworth low-pass digital filter.

    Parameters:
        waveform (np.ndarray): Raw PPG signal waveform.
        order (int): Filter order (should be higher than zero).
        lp (float): Low-pass frequency.
        fs (float): Sample frequency of the signal (Hz).

    Returns:
        np.ndarray: Low-pass filtered PPG signal waveform.
    """

    if order < 2:
        raise ValueError('Order must to be higher than zero')

    b, a = signal.butter(order, lp, btype='lowpass', fs=fs)
    return signal.lfilter(b, a, waveform - waveform[0])


def baseline_correction(waveform: np.ndarray) -> np.ndarray:
    """
    Baseline correction using adaptive iteratively reweighted penalized least squares.

    Parameters:
        waveform (np.ndarray): Raw PPG signal waveform.

    Return:
        np.ndarray: Baseline-corrected signal.

    Reference:
    Zhang ZM, Chen S, Liang YZ. Baseline correction using adaptive iteratively reweighted
    penalized least squares. Analyst. 2010 May;135(5):1138-46. doi: 10.1039/b922045c.
    Epub 2010 Feb 19. PMID: 20419267
    """
    return BaselineRemoval(waveform).ZhangFit()
