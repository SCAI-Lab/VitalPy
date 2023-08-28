"""
Some code has been extracted from Time Series Feature Extraction Library
https://github.com/ElsevierSoftwareX/SOFTX_2020_1

Reference:
M. Barandas, D. Folgado, L. Fernandes, S. Santos, M. Abreu, P. Bota, H. Liu, T. Schultz, H. Gamboa
Tsfel: Time Series Feature Extraction Library
SoftwareX, 11 (2020), p. 100456
"""

import scipy
import numpy as np
from biosppy.signals import ecg

N_SEG = 200  # biosppy.signals.ecg.fSQI input parameter


def frequency_first(signal: np.ndarray, fs: int) -> float:
    """
    Calculates the frequency of the first peak in the signal's FFT.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The frequency of the first peak in the signal's FFT.
    """
    peak = 1  # first peak
    f_base, sp_mag_base = get_fft_peaks(signal, fs, peak)
    return f_base


def mag_first(signal: np.ndarray, fs: int) -> float:
    """
    Calculates the magnitude of the first peak in the signal's FFT.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The magnitude of the first peak in the signal's FFT.
    """
    peak = 1  # first peak
    f_base, sp_mag_base = get_fft_peaks(signal, fs, peak)
    return sp_mag_base


def frequency_second(signal: np.ndarray, fs: int) -> float:
    """
    Calculates the frequency of the second peak in the signal's FFT.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The frequency of the second peak in the signal's FFT.
    """
    peak = 2  # second peak
    f_base, sp_mag_base = get_fft_peaks(signal, fs, peak)
    return f_base


def mag_second(signal: np.ndarray, fs: int):
    """
    Calculates the magnitude of the second peak in the signal's FFT.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The magnitude of the second peak in the signal's FFT.
    """
    peak = 2  # second peak
    f_base, sp_mag_base = get_fft_peaks(signal, fs, peak)
    return sp_mag_base


def frequency_third(signal: np.ndarray, fs: int):
    """
    Calculates the frequency of the third peak in the signal's FFT.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The frequency of the third peak in the signal's FFT.
    """
    peak = 3  # third peak
    f_base, sp_mag_base = get_fft_peaks(signal, fs, peak)
    return f_base


def mag_third(signal: np.ndarray, fs: int):
    """
    Calculates the magnitude of the third peak in the signal's FFT.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The magnitude of the third peak in the signal's FFT.
    """
    peak = 3  # third peak
    f_base, sp_mag_base = get_fft_peaks(signal, fs, peak)
    return sp_mag_base


def fsqi_first(signal: np.ndarray, fs: int):
    """
    Return the ratio between two frequency bands:
    The frequency bandwidth for the ratio's numerator: from 0 to f1
    The frequency bandwidth for the ratio's denominator is the whole spectrum

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The fSQI (Frequency Signal Quality Index) value for the first frequency band.
    """
    f1 = frequency_first(signal, fs)
    return ecg.fSQI(signal, fs, nseg=N_SEG, num_spectrum=[0, f1])


def fsqi_second(signal: np.ndarray, fs: int):
    """
    Return the ratio between two frequency bands:
    The frequency bandwidth for the ratio's numerator: from 0 to f2
    The frequency bandwidth for the ratio's denominator is the whole spectrum

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The fSQI (Frequency Signal Quality Index) value for the second frequency band.
    """
    f2 = frequency_second(signal, fs)
    return ecg.fSQI(signal, fs, nseg=N_SEG, num_spectrum=[0, f2])


def fsqi_third(signal: np.ndarray, fs: int):
    """
    Return the ratio between two frequency bands:
    The frequency bandwidth for the ratio's numerator: from 0 to f3
    The frequency bandwidth for the ratio's denominator is the whole spectrum

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The fSQI (Frequency Signal Quality Index) value for the third frequency band.
    """
    f3 = frequency_first(signal, fs)
    return ecg.fSQI(signal, fs, nseg=N_SEG, num_spectrum=[0, f3])


def fsqi(signal: np.ndarray, fs: int):
    """
    Return the ratio between two frequency bands:
    The frequency bandwidth for the ratios' numerator: from 1 to 2.25 Hz
    The frequency bandwidth for the ratio's denominator: from 0 to 8 Hz

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The fSQI (Frequency Signal Quality Index) value for the frequency band.
    """
    return ecg.fSQI(signal, fs, nseg=N_SEG, num_spectrum=[1, 2.25], dem_spectrum=[0, 8])


def spectral_distance(signal: np.ndarray, fs: int):
    """
    Calculates the distance of the signal's cumulative sum of the FFT elements to
    the respective linear regression.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The spectral distance value.
    """
    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)

    # Linear regression
    points_y = np.linspace(0, cum_fmag[-1], len(cum_fmag))
    return np.sum(points_y - cum_fmag)


def fundamental_frequency(signal: np.ndarray, fs: int):
    """
    Calculates the predominant frequency of the signal.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The fundamental frequency of the signal.
    """

    signal = signal - np.mean(signal)
    f, fmag = calc_fft(signal, fs)

    # Condition for offset removal, since the offset generates a peak at frequency zero
    try:
        # With 0.1 the offset frequency is discarded
        cond = np.where(f > 0.1)[0][0]
    except IndexError:
        cond = 0

    # Finding big peaks, not considering noise peaks with low amplitude
    bp = scipy.signal.find_peaks(fmag[cond:], threshold=10)[0]
    if not list(bp):
        f0 = 0
    else:
        # add cond for correcting the indices position
        bp = bp + cond
        # f0 is the minimum big peak frequency
        f0 = f[min(bp)]

    return f0


def max_power_spectrum(signal: np.ndarray, fs: int):
    """
    Calculates the maximum value of the power spectrum density of the signal.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The maximum value of the power spectrum density.
    """
    if np.std(signal) == 0:
        return float(max(scipy.signal.welch(signal, int(fs), nperseg=len(signal))[1]))
    else:
        return float(max(scipy.signal.welch(signal / np.std(signal), int(fs), nperseg=len(signal))[1]))


def max_frequency(signal: np.ndarray, fs: int):
    """
    Calculates 0.95 of maximum frequency of the signal using cumsum.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The maximum frequency of the signal with the specified cumulative cutoff.
    """

    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)

    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.95)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)
    return f[ind_mag]


def median_frequency(signal: np.ndarray, fs: int):
    """
    Calculates the median frequency of the signal.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The median frequency of the signal.
    """

    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)
    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.50)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)
    f_median = f[ind_mag]
    return f_median


def spectral_centroid(signal: np.ndarray, fs: int):
    """
    Calculates the barycenter of the spectrum (spectral centroid).

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The spectral centroid of the signal.
    """

    f, fmag = calc_fft(signal, fs)
    if not np.sum(fmag):
        return 0
    else:
        return np.dot(f, fmag / np.sum(fmag))


def spectral_decrease(signal: np.ndarray, fs: int):
    """
    Measure for the decrease of the spectral amplitude.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The spectral decrease of the signal.
    """

    f, fmag = calc_fft(signal, fs)

    fmag_band = fmag[1:]
    len_fmag_band = np.arange(2, len(fmag) + 1)

    # Sum of numerator
    soma_num = np.sum((fmag_band - fmag[0]) / (len_fmag_band - 1), axis=0)

    if not np.sum(fmag_band):
        return 0
    else:
        # Sum of denominator
        soma_den = 1 / np.sum(fmag_band)

        # Spectral decrease computing
        return soma_den * soma_num


def spectral_kurtosis(signal: np.ndarray, fs: int):
    """
    Measure for 'the flatness of a distribution around its mean value'.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The spectral kurtosis of the signal.
    """

    f, fmag = calc_fft(signal, fs)
    if not spectral_spread(signal, fs):
        return 0
    else:
        spect_kurt = ((f - spectral_centroid(signal, fs)) ** 4) * (fmag / np.sum(fmag))
        return np.sum(spect_kurt) / (spectral_spread(signal, fs) ** 4)


def spectral_skewness(signal: np.ndarray, fs: int):
    """
    Measures the asymmetry of a distribution around its mean value.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The spectral skewness of the signal.
    """

    f, fmag = calc_fft(signal, fs)
    spect_centr = spectral_centroid(signal, fs)

    if not spectral_spread(signal, fs):
        return 0
    else:
        skew = ((f - spect_centr) ** 3) * (fmag / np.sum(fmag))
        return np.sum(skew) / (spectral_spread(signal, fs) ** 3)


def spectral_spread(signal: np.ndarray, fs: int):
    """
    Measures the spread of the spectrum around its mean value.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The spectral spread of the signal.
    """

    f, fmag = calc_fft(signal, fs)
    spect_centroid = spectral_centroid(signal, fs)

    if not np.sum(fmag):
        return 0
    else:
        return np.dot(((f - spect_centroid) ** 2), (fmag / np.sum(fmag))) ** 0.5


def spectral_slope(signal: np.ndarray, fs: int):
    """
    Computes the spectral slope.
    Spectral slope is computed by finding constants m and b of the function
    aFFT = mf + b, obtained by linear regression of the spectral amplitude.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The spectral slope of the signal.
    """

    f, fmag = calc_fft(signal, fs)

    if not (list(f)) or (np.sum(fmag) == 0):
        return 0
    else:
        if not (len(f) * np.dot(f, f) - np.sum(f) ** 2):
            return 0
        else:
            num_ = (1 / np.sum(fmag)) * (len(f) * np.dot(f, fmag) - np.sum(f) * np.sum(fmag))
            denom_ = (len(f) * np.dot(f, f) - np.sum(f) ** 2)
            return num_ / denom_


def spectral_variation(signal: np.ndarray, fs: int):
    """
    Computes the amount of variation of the spectrum along the temporal dimension.
    Spectral variation is computed from the normalized cross-correlation between two consecutive amplitude spectra.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The spectral variation of the signal.
    """

    f, fmag = calc_fft(signal, fs)

    sum1 = np.sum(np.array(fmag)[:-1] * np.array(fmag)[1:])
    sum2 = np.sum(np.array(fmag)[1:] ** 2)
    sum3 = np.sum(np.array(fmag)[:-1] ** 2)

    if not sum2 or not sum3:
        variation = 1
    else:
        variation = 1 - (sum1 / ((sum2 ** 0.5) * (sum3 ** 0.5)))

    return variation


def spectral_maxpeaks(signal: np.ndarray, fs: int):
    """
    Counts the number of maximum spectral peaks of the signal.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        int: The number of maximum spectral peaks.
    """

    f, fmag = calc_fft(signal, fs)
    diff_sig = np.diff(fmag)

    return np.sum([1 for nd in range(len(diff_sig[:-1])) if ((diff_sig[nd + 1] < 0) and (diff_sig[nd] > 0))])


def spectral_roll_off(signal: np.ndarray, fs: int):
    """
    Computes the spectral roll-off of the signal.
    The spectral roll-off corresponds to the frequency where 95% of the signal magnitude is contained below this value.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The spectral roll-off of the signal.
    """

    f, fmag = calc_fft(signal, fs)
    cum_ff = np.cumsum(fmag)
    value = 0.95 * (np.sum(fmag))

    return f[np.where(cum_ff >= value)[0][0]]


def spectral_roll_on(signal: np.ndarray, fs: int):
    """
    Computes the spectral roll-on of the signal.
    The spectral roll-on corresponds to the frequency
    where 5% of the signal magnitude is contained
    below this value.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The spectral roll-on of the signal.
    """

    f, fmag = calc_fft(signal, fs)
    cum_ff = np.cumsum(fmag)
    value = 0.05 * (np.sum(fmag))

    return f[np.where(cum_ff >= value)[0][0]]


def human_range_energy(signal: np.ndarray, fs: int):
    """
    Computes the human range energy ratio.
    The human range energy ratio is given by the ratio between the energy
    in the frequency range of 0.6-2.5Hz and the whole energy band.

    Args:
        signal (np.array): The input signal.
        fs (int): The sampling rate of the signal.

    Returns:
        float: The human range energy ratio.
    """

    f, fmag = calc_fft(signal, fs)

    allenergy = np.sum(fmag ** 2)

    if allenergy == 0:
        # For handling the occurrence of Nan values
        return 0.0

    hr_energy = np.sum(fmag[np.argmin(np.abs(0.6 - f)):np.argmin(np.abs(2.5 - f))] ** 2)

    ratio = hr_energy / allenergy

    return ratio


def power_bandwidth(signal: np.ndarray, fs: int):
    """Computes power spectrum density bandwidth of the signal.
      It corresponds to the width of the frequency band in which 95% of its power is located.
      Description in article:
      Power Spectrum and Bandwidth Ulf Henriksson, 2003 Translated by Mikael Olofsson, 2005

    Args:
       signal (np.array): Input from which the power bandwidth computed
       fs (float): Sampling frequency of the input signal.

    Returns:
        float: Occupied power in bandwidth
    """

    # Computing the power spectrum density
    if np.std(signal) == 0:
        freq, power = scipy.signal.welch(signal, fs, nperseg=len(signal))
    else:
        freq, power = scipy.signal.welch(signal / np.std(signal), fs, nperseg=len(signal))

    if np.sum(power) == 0:
        return 0.0

    # Computing the lower and upper limits of power bandwidth
    cum_power = np.cumsum(power)
    f_lower = freq[np.where(cum_power >= cum_power[-1] * 0.95)[0][0]]

    cum_power_inv = np.cumsum(power[::-1])
    f_upper = freq[np.abs(np.where(cum_power_inv >= cum_power[-1] * 0.95)[0][0] - len(power) + 1)]

    # Returning the bandwidth in terms of frequency

    return np.abs(f_upper - f_lower)


def fft_mean_coeff(signal: np.ndarray, fs: int, nfreq=256):
    """Computes the mean value of each spectrogram frequency.
       nfreq can not be higher than half signal length plus one.
       When it does, it is automatically set to half signal length plus one.


    Args:
        signal (np.array): Input from which fft mean coefficients are computed
        fs (float): Sampling frequency of the input signal.
        nfreq (int): The number of frequencies

    Returns:
    np.array:  The mean value of each spectrogram frequency
    """

    if nfreq > len(signal) // 2 + 1:
        nfreq = len(signal) // 2 + 1

    fmag_mean = scipy.signal.spectrogram(signal, fs, nperseg=nfreq * 2 - 2)[2].mean(1)

    return tuple(fmag_mean)


def spectral_entropy(signal: np.ndarray, fs: int):
    """Computes the spectral entropy of the signal based on Fourier transform.

    Args:
        signal (np.array): Input from which spectral entropy is computed
        fs (float): Sampling frequency of the input signal.

    Returns:
        float: The normalized spectral entropy value
    """
    # Removing DC component
    sig = signal - np.mean(signal)

    f, fmag = calc_fft(sig, fs)

    power = fmag ** 2

    if power.sum() == 0:
        return 0.0

    prob = np.divide(power, power.sum())

    prob = prob[prob != 0]

    return -np.multiply(prob, np.log2(prob)).sum() / np.log2(prob.size)


def wavelet_entropy(signal: np.ndarray, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """ Computes CWT entropy of the signal.
        Implementation details in:
        https://dsp.stackexchange.com/questions/13055/how-to-calculate-cwt-shannon-entropy
        B.F. Yan, A. Miyamoto, E. Bruhwiler, Wavelet transform-based modal parameter identification considering uncertainty

    Args:
        signal (np.array): Input from which CWT is computed
        function :  wavelet function, Defaults to scipy.signal.ricker
        widths (np.array): Widths to use for transformation, Defaults to np.arange(1,10)

    Returns:
        float: wavelet entropy
    """
    if np.sum(signal) == 0:
        return 0.0

    cwt = wavelet(signal, function, widths)
    energy_scale = np.sum(np.abs(cwt), axis=1)
    t_energy = np.sum(energy_scale)
    prob = energy_scale / t_energy
    w_entropy = -np.sum(prob * np.log(prob))

    return w_entropy


def wavelet_abs_mean(signal: np.ndarray, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT absolute mean value of each wavelet scale.

    Args:
        signal (np.array): Input from which CWT is computed
        function :  wavelet function, Defaults to scipy.signal.ricker
        widths (np.array): Widths to use for transformation, Defaults to np.arange(1,10)

    Returns:
        tuple: CWT absolute mean value
    """

    return tuple(np.abs(np.mean(wavelet(signal, function, widths), axis=1)))


def wavelet_std(signal: np.ndarray, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT std value of each wavelet scale.

    Args:
        signal (np.array): Input from which CWT is computed
        function :  wavelet function, Defaults to scipy.signal.ricker
        widths (np.array): Widths to use for transformation, Defaults to np.arange(1,10)

    Returns:
        tuple: CWT std
    """

    return tuple((np.std(wavelet(signal, function, widths), axis=1)))


def wavelet_var(signal: np.ndarray, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT variance value of each wavelet scale.

    Args:
        signal (np.array): Input from which CWT is computed
        function :  wavelet function, Defaults to scipy.signal.ricker
        widths (np.array): Widths to use for transformation, Defaults to np.arange(1,10)

    Returns:
        tuple: CWT variance
    """

    return tuple((np.var(wavelet(signal, function, widths), axis=1)))


# COMPLEMENTARY FUNCTIONS

def wavelet_energy(signal: np.ndarray, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """ Computes CWT energy of each wavelet scale.
        Implementation details:
        https://stackoverflow.com/questions/37659422/energy-for-1-d-wavelet-in-python

    Args:
        signal (np.array): Input from which CWT is computed
        function :  wavelet function, Defaults to scipy.signal.ricker
        widths (np.array): Widths to use for transformation, Defaults to np.arange(1,10)

    Returns:
        tuple: CWT energy
    """

    cwt = wavelet(signal, function, widths)
    energy = np.sqrt(np.sum(cwt ** 2, axis=1) / np.shape(cwt)[1])

    return tuple(energy)


def filterbank(signal: np.ndarray, fs, pre_emphasis=0.97, nfft=512, nfilt=40):
    """ Computes the MEL-spaced filterbank.
        It provides the information about the power in each frequency band.
        Implementation details and description on:
        https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
        https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1

    Args:
        signal (np.array): Input from which filterbank is computed
        fs (float): Sampling frequency of the input signal.
        pre_emphasis (float): Pre-emphasis coefficient for pre-emphasis filter application
        nfft (int): Number of points of fft
        nfilt (int): Number of filters

    Returns:
    np.array: MEL-spaced filterbank
    """

    # Signal is already a window from the original signal, so no frame is needed.
    # According to the references it is needed the application of a window function such as
    # hann window. However, if the signal windows don't have overlap, we will lose information,
    # as the application of a hann window will overshadow the windows signal edges.

    # pre-emphasis filter to amplify the high frequencies

    emphasized_signal = np.append(np.array(signal)[0], np.array(signal[1:]) - pre_emphasis * np.array(signal[:-1]))

    # Fourier transform and Power spectrum
    mag_frames = np.absolute(np.fft.rfft(emphasized_signal, nfft))  # Magnitude of the FFT

    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    filter_bin = np.floor((nfft + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):

        f_m_minus = int(filter_bin[m - 1])  # left
        f_m = int(filter_bin[m])  # center
        f_m_plus = int(filter_bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - filter_bin[m - 1]) / (filter_bin[m] - filter_bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (filter_bin[m + 1] - k) / (filter_bin[m + 1] - filter_bin[m])

    # Area Normalization
    # If we don't normalize the noise will increase with frequency because of the filter width.
    enorm = 2.0 / (hz_points[2:nfilt + 2] - hz_points[:nfilt])
    fbank *= enorm[:, np.newaxis]

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    return filter_banks


def wavelet(signal: np.ndarray, function=scipy.signal.ricker, widths=np.arange(1, 10)):
    """Computes CWT (continuous wavelet transform) of the signal.

    Args:
        signal (np.array): Input from which CWT is computed
        function :  wavelet function, Defaults to scipy.signal.ricker
        widths (np.array): Widths to use for transformation, Defaults to np.arange(1,10)

    Returns:
        np.array: The result of the CWT along the time axis matrix with size (len(widths),len(signal))
    """

    if isinstance(function, str):
        function = eval(function)

    if isinstance(widths, str):
        widths = eval(widths)

    cwt = scipy.signal.cwt(signal, function, widths)

    return cwt


def lpc(signal: np.ndarray, n_coeff=12):
    """ Computes the linear prediction coefficients.
        Implementation details and description in:
        https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Args:
        signal (np.array): Input from linear prediction coefficients are computed
        n_coeff (int): Number of coefficients

    Returns:
        np.array: Linear prediction coefficients
    """

    # Calculate LPC with Yule-Walker
    acf = autocorr_norm(signal)
    r = -acf[1:n_coeff + 1].T
    smatrix = create_symmetric_matrix(acf, n_coeff)
    if np.sum(smatrix) == 0:
        return tuple(np.zeros(n_coeff))

    lpc_coeffs = np.dot(np.linalg.inv(smatrix), r)
    lpc_coeffs = lpc_coeffs / np.max(np.abs(lpc_coeffs))

    return tuple(lpc_coeffs)


def autocorr_norm(signal: np.ndarray):
    """ Computes the autocorrelation.
        Implementation details and description in:
        https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Args:
        signal (np.array): Input from linear prediction coefficients are computed

    Returns:
        np.array: Autocorrelation result
    """

    signal = signal - np.mean(signal)
    r = np.correlate(signal, signal, mode='full')[-len(signal):]

    if np.sum(signal) == 0:
        return np.zeros(len(signal))

    acf = r / (np.var(signal) * (np.arange(len(signal), 0, -1)))

    return acf


def create_symmetric_matrix(acf, n_coeff=12):
    """ Computes a symmetric matrix.
        Implementation details and description in:
        https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Args:
        acf (np.array): Input from which a symmetric matrix is computed
        n_coeff (int): Number of coefficients

    Returns:
        np.array: Symmetric Matrix
    """

    smatrix = np.empty((n_coeff, n_coeff))

    for i in range(n_coeff):
        for j in range(n_coeff):
            smatrix[i, j] = acf[np.abs(i - j)]
    return smatrix


def calc_fft(signal: np.ndarray, fs: int):
    """ This functions computes the fft of a signal.

    Args:
        signal (np.array): The input signal for which the fft is computed
        fs (float): The sampling frequency of the waveform in Hz.

    Returns (tuple):
        - np.array: Frequency values (xx axis)
        - np.array: Amplitude of the frequency values (yy axis)
    """

    fmag = np.abs(np.fft.fft(signal))
    f = np.linspace(0, fs // 2, len(signal) // 2)

    return f[:len(signal) // 2].copy(), fmag[:len(signal) // 2].copy()


def __next_pow2(x):
    return 1 << (x - 1).bit_length()


def get_fft_peaks(signal: np.ndarray, fs: int, peak: int):
    """Calculates the frequency and magnitude of the nth highest peak in the FFT of a signal.

    Args:
        signal (np.array): Input signal in time domain
        fs (float): The sampling frequency of the waveform in Hz.
        peak (int): Nth highest peak to be extracted

    Returns:
        Tuple[float, float]: The frequency and magnitude of the nth highest peak
    """

    sp_mag = np.abs(np.fft.fft(signal, n=__next_pow2(len(signal)) * 16))
    freqs = np.fft.fftfreq(len(sp_mag))
    # calculate the peaks
    sp_mag_maxima_index = scipy.signal.argrelmax(sp_mag)[0]
    num = peak - 1
    # f_base
    f_x = freqs[sp_mag_maxima_index[num]] * fs
    # sp_mag_base
    sp_mag_x = sp_mag[sp_mag_maxima_index[num]] / len(signal)

    return f_x, sp_mag_x
