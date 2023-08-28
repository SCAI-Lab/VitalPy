"""
Some code has been extracted from the Time Series Feature Extraction Library:
https://github.com/ElsevierSoftwareX/SOFTX_2020_1

Reference:
M. Barandas, D. Folgado, L. Fernandes, S. Santos, M. Abreu, P. Bota, H. Liu, T. Schultz, H. Gamboa
Tsfel: Time Series Feature Extraction Library
SoftwareX, 11 (2020), p. 100456
"""

import numpy as np
import scipy
from ...ppg import utils


def autocorr(signal: np.ndarray) -> float:
    return float(np.correlate(signal, signal))


def calc_centroid(signal, fs):
    # Computes the centroid
    time = utils.timeline(signal, fs)

    energy = np.array(signal) ** 2

    t_energy = np.dot(time, energy)
    energy_sum = np.sum(energy)

    if energy_sum == 0 or t_energy == 0:
        centroid = 0
    else:
        centroid = t_energy / energy_sum
    return centroid


def entropy_kde(signal: np.ndarray) -> float:
    signal_ = np.copy(signal)
    # Computes the probability density function of the input signal using a Gaussian KDE (Kernel Density Estimate)
    if min(signal_) == max(signal_):
        noise = np.random.randn(len(signal_)) * 0.0001
        signal_ = np.copy(signal_ + noise)

    kernel = scipy.stats.gaussian_kde(signal_, bw_method='silverman')

    p = np.array(kernel(signal_) / np.sum(kernel(signal_)))

    return entropy(signal_, p)


def entropy_gauss(signal: np.ndarray) -> float:
    # Computes the probability density function of the input signal using a Gaussian function
    signal_ = np.copy(signal)

    std_value = np.std(signal_)
    mean_value = np.mean(signal_)

    if std_value == 0:
        p = 0.0
    else:
        pdf_gauss = scipy.stats.norm.pdf(signal_, mean_value, std_value)
        p = np.array(pdf_gauss / np.sum(pdf_gauss))

    return entropy(signal_, p)


def minpeaks(signal: np.ndarray):
    # Computes number of minimum peaks of the signal.
    diff_sig = np.diff(signal)
    return np.sum([1 for nd in range(len(diff_sig[:-1])) if ((diff_sig[nd] < 0) and (diff_sig[nd + 1] > 0))])


def maxpeaks(signal: np.ndarray):
    # Computes number of maximum peaks of the signal.
    diff_sig = np.diff(signal)
    return np.sum([1 for nd in range(len(diff_sig[:-1])) if ((diff_sig[nd + 1] < 0) and (diff_sig[nd] > 0))])


def mean_abs_diff(signal: np.ndarray):
    # Computes mean absolute differences of the signal.
    return np.mean(np.abs(np.diff(signal)))


def mean_diff(signal: np.ndarray):
    # Computes mean of differences of the signal.
    return np.mean(np.diff(signal))


def median_abs_diff(signal):
    # Computes median absolute differences of the signal.
    return np.median(np.abs(np.diff(signal)))


def median_diff(signal: np.ndarray):
    # Compute median of differences of the signal
    return np.median(np.diff(signal))


def distance(signal: np.ndarray):
    # Computes signal traveled distance using the hipotenusa between 2 points
    diff_sig = np.diff(signal)
    return np.sum([np.sqrt(1 + df ** 2) for df in diff_sig])


def sum_abs_diff(signal: np.ndarray):
    return np.sum(np.abs(np.diff(signal)))


def zero_cross_1d(signal: np.ndarray, fs: float):
    # Number of times that the first derivative cross zero
    derivative1 = utils.derivative(signal, fs)
    return utils.zero_cross(derivative1)


def zero_cross_2d(signal: np.ndarray, fs: float):
    # Number of times that the first derivative cross zero
    derivative1 = utils.derivative(signal, fs)
    derivative2 = utils.derivative(derivative1, fs)
    return utils.zero_cross(derivative2)


def zero_cross_3d(signal: np.ndarray, fs: float):
    # Number of times that the first derivative cross zero
    derivative1 = utils.derivative(signal, fs)
    derivative2 = utils.derivative(derivative1, fs)
    derivative3 = utils.derivative(derivative2, fs)
    return utils.zero_cross(derivative3)


def total_energy(signal: np.ndarray, fs: float):
    time = utils.timeline(signal, fs)
    return np.sum(np.array(signal) ** 2) / (time[-1] - time[0])


def slope(signal: np.ndarray, fs: float):
    # Computes the slope of the signal.
    time = utils.timeline(signal, fs)
    return np.polyfit(time, signal, 1)[0]


def abs_energy(signal: np.ndarray):
    return np.sum(signal ** 2)


# COMMON FUNCTIONS
def entropy(signal: np.ndarray, prob: np.ndarray):
    # Calculate entropy
    if np.sum(prob) == 0:
        return 0.0

    # Handling zero probability values
    p = prob[np.where(prob != 0)]

    if np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
        return 0.0
    else:
        return - np.sum(p * np.log2(p)) / np.log2(len(signal))
