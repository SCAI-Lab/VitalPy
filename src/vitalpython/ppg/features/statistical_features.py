import scipy
import numpy as np


def skewness(signal: np.ndarray):
    """ Calculate the skewness (S) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Skewness
    """

    return scipy.stats.skew(signal)


def kurtosis(signal: np.ndarray):
    """ Calculate the kurtosis (K) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Kurtosis
    """
    return scipy.stats.kurtosis(signal)


def mav(signal: np.ndarray):
    """ Calculate the Mean Absolute Value (MAV) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Mean Absolute Value
    """

    return np.mean(abs(signal))


def median(signal: np.ndarray):
    """ Calculate the median of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Median
    """
    return np.median(signal)


def mad(signal: np.ndarray):
    """ Calculate the Mean Absolute Deviation (MAD) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Mean Absolute Deviation
    """
    return np.mean(np.abs(signal - np.mean(signal, axis=0)), axis=0)


def med_ad(signal: np.ndarray):
    """ Calculate the Median absolute deviation (mAD) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Median Absolute Deviation
    """
    return scipy.stats.median_abs_deviation(signal, scale=1)


def rms(signal: np.ndarray):
    """ Calculate the Root-Mean-Square (RMS) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Root-Mean-Square
    """
    return np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))


def sd(signal: np.ndarray):
    """ Calculate the Standard Deviation (SD) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Standard Deviation
    """
    return np.std(signal)


def shape_factor(signal: np.ndarray):
    """ Calculate the Shape Factor (SF) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Shape Factor
    """
    rms = np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))
    mav = np.mean(abs(signal))
    return rms / mav


def impulse_factor(signal: np.ndarray):
    """ Calculate the Impulse Factor (IF) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Impulse Factor
    """
    return max(signal) / np.mean(signal)


def crest_factor(signal: np.ndarray):
    """ Calculate the Crest Factor (CF) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Crest Factor
    """
    rms = np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))
    return max(signal) / rms


def variance(signal: np.ndarray):
    """ Calculate the variance of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Variance
    """
    return np.var(signal)


def irq(signal: np.ndarray):
    """ Calculate the Interquartile Range (IRQ) of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        Interquartile Range
    """
    return scipy.stats.iqr(signal)


def perfussion(signal: np.ndarray):
    """ Calculate the perfussion of the signal

    Args:
        signal (np.array): data points of the beat-to-beat or segment of the PPG signal

    Returns:
        perfussion
    """
    return (max(signal) - min(signal)) / np.mean(signal)



