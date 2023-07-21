import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import argrelmax, argrelmin
from src.ppg.params import PPG_PEAK_DETECTION_THRESHOLD


def find_extremum(waveform: pd.Series, show: bool = False):
    """
    Find all the extrema (minima and maxima) in the waveform.

    Parameters:
        waveform (pd.Series): The input waveform as a pandas Series.
        show (bool, optional): Flag for showing the results. Defaults to False.

    Returns:
        list: A list of tuples containing the indices and values of the extrema.
    """

    wave_np = np.array(waveform)
    extremum_idx = np.sort(np.unique(np.concatenate((argrelmax(wave_np)[0], argrelmin(wave_np)[0]))))
    extremum = wave_np[extremum_idx]

    if show:
        plt.plot(wave_np)
        plt.plot(extremum_idx, extremum, 'o')
        plt.title('All possible extremas')

    return zip(extremum_idx.tolist(), extremum.tolist())


def segmentation(waveform: pd.Series):
    """
    Pulse segmentation or extraction of all beat-to-beat single waveforms (templates).
    Pulse segmentation is done by analyzing the relative extrema of the signal using an adaptive pulse amplitude threshold.

    Parameters:
        waveform (pd.Series): PPG waveform as a pandas Series.

    Return:
        list: List of beat-to-beat single waveform templates (pd.Series).
    """

    template_list = []

    wave_np = waveform.to_numpy()
    max_value = np.median(wave_np[argrelmax(wave_np)])
    min_value = np.median(wave_np[argrelmin(wave_np)])
    threshold = (max_value - min_value) * PPG_PEAK_DETECTION_THRESHOLD

    last_extreme_index = None
    last_extreme = None
    last_single_waveform_start_index = None
    for extreme_index, extreme in find_extremum(waveform):
        if last_extreme is not None and extreme - last_extreme > threshold:
            if last_single_waveform_start_index is not None:
                single_waveform = waveform[last_single_waveform_start_index:last_extreme_index]
                template_list.append(single_waveform)
            last_single_waveform_start_index = last_extreme_index
        last_extreme_index = extreme_index
        last_extreme = extreme

    return template_list
