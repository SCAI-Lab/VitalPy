import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
import src.ppg.utils as utils
from src.ppg.params import MIN_MAX_EUC_DIST_THRESHOLD, PPG_SAMPLE_RATE, Z_SCORE_EUC_DIST_THRESHOLD


def euclidian_analysis(valid_data: list, fs: float = PPG_SAMPLE_RATE, norm_method: str = 'zscore',
                       distance_threshold: float = None, show: bool = False):
    """
    Perform Euclidian analysis on the valid_data waveforms.

    Parameters:
        valid_data (list): List of beat-to-beat single waveform templates (pd.Series).
        fs (float, optional): Sample frequency of the signal (Hz). Defaults to PPG_SAMPLE_RATE.
        norm_method (str, optional): Normalization method. Either 'min_max' or 'zscore'. Defaults to 'zscore'.
        distance_threshold (float, optional): Euclidian distance threshold for checking similarity.
                                              Defaults to None (uses predefined thresholds based on norm_method).
        show (bool, optional): Whether to display plots. Defaults to False.

    Returns:
        Tuple[list, pd.Series]: A tuple containing:
                                - euc_valid_waveforms (list): List of beat-to-beat waveforms that meet the similarity threshold.
                                - mean_waveform_pd (pd.Series): The mean waveform calculated from the valid waveforms.
    """
    waveforms = []

    # define default thresholds
    if distance_threshold is None:
        if norm_method == 'min_max':
            distance_threshold = MIN_MAX_EUC_DIST_THRESHOLD
        elif norm_method == 'zscore':
            distance_threshold = Z_SCORE_EUC_DIST_THRESHOLD

    # calculate max length and max peak of the valid data
    peak_idx = []
    len_wave = []
    for data in valid_data:
        peak_idx.append(np.argmax(data))
        len_wave.append(len(data))

    # get maximum and minimum values
    max_peak_idx = int(np.max(peak_idx))
    max_len_wave = int(np.max(len_wave))

    # for each single valid waveform
    for pk, data in zip(peak_idx, valid_data):
        data_np = data.to_numpy()

        # shift the signal so that the systolic peak is at max_peak_idx and it has a length = max_len_wave
        # fill the rest with nan values
        if pk >= max_peak_idx:
            wave_shifted = np.array(data_np[pk - max_peak_idx:pk + max_len_wave - max_peak_idx])
        else:
            dif_peak = int(max_peak_idx - pk)
            init_part = np.full((dif_peak,), np.nan)
            middle_part = np.array(data_np[0:pk + max_len_wave - max_peak_idx])
            wave_shifted = np.concatenate((init_part, middle_part))
        end_part = np.full((int(max_len_wave - len(wave_shifted)),), np.nan)
        wave_shifted = np.concatenate((wave_shifted, end_part))

        # normalize shifted signal and save in list
        try:
            if norm_method == 'min_max':
                wave_n = (wave_shifted - np.nanmin(wave_shifted)) / (np.nanmax(wave_shifted) - np.nanmin(wave_shifted))
            elif norm_method == 'zscore':
                wave_n = stats.zscore(wave_shifted, nan_policy='omit')
            else:
                raise ValueError('norm_method argument not valid. It only accept min_max and z_score. Got %s',
                                 norm_method)
        except:
            print("An exception occurred")
        else:
            waveforms.append(wave_n)

    # calculate the mean of the waveforms. Hence, it is also shifted and normalized
    mean_waveform = np.nanmean(waveforms, axis=0)

    euc_dist = []  # to save euclidian distance
    euc_valid_waveforms = []  # to save the valid and original pd.series waveforms (index = time)
    euc_valid_plot = []  # to save the valid and shifted waveforms (only for plotting)
    for wave, vdata in zip(waveforms, valid_data):
        dist = np.sqrt(np.nansum((wave - mean_waveform) ** 2))
        euc_dist.append(dist)
        if dist < distance_threshold:
            euc_valid_waveforms.append(vdata)
            euc_valid_plot.append(wave)

    # generate a pd.Series with the mean waveform calculated from only the valid (euclidian distance checking) waves
    mean_waveform_pd = pd.Series()

    # if there are at least one good waveforms
    if len(euc_valid_waveforms) > 1:
        # recalculate everything to obtain the mean
        re_peak_idx = []
        re_len_wave = []
        for wave in euc_valid_waveforms:
            re_peak_idx.append(np.argmax(wave))
            re_len_wave.append(len(wave))
        mean_repeak_idx = int(np.mean(re_peak_idx))
        mean_relen_wave = int(np.mean(re_len_wave))

        re_mean_waveform = np.nanmean(euc_valid_plot, axis=0)
        cut_mean_wave = re_mean_waveform[max_peak_idx - mean_repeak_idx:-1]
        cut_mean_wave = cut_mean_wave[0:mean_relen_wave]
        mean_waveform_pd = pd.Series(cut_mean_wave, index=utils.timeline(cut_mean_wave, fs))

    if show:
        if len(euc_valid_waveforms) == 0:
            # only show the disaster
            time = utils.timeline(mean_waveform, fs)
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)
            for wave in waveforms:
                ax.plot(time, wave, color='lightblue', label='waveforms')
            ax.plot(time, mean_waveform, color='royalblue', linewidth=2, label='Mean')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('PPG (a.u.)')
            ax.set_title("All waveforms and mean")
            # Get artists and labels for legend and chose which ones to display
            handles, labels = ax.get_legend_handles_labels()
            display = (0, len(waveforms))
            ax.legend([handle for i, handle in enumerate(handles) if i in display],
                      [label for i, label in enumerate(labels) if i in display], loc='best')

        else:
            time = utils.timeline(mean_waveform, fs)
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(1, 2, 1)
            for wave in waveforms:
                ax.plot(time, wave, color='lightblue', label='waveforms')
            ax.plot(time, mean_waveform, color='royalblue', linewidth=2, label='Mean')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('PPG (a.u.)')
            ax.set_title("Similarity checking - Signals shifted to systolic peak")
            # Get artists and labels for legend and chose which ones to display
            handles, labels = ax.get_legend_handles_labels()
            display = (0, len(waveforms))
            ax.legend([handle for i, handle in enumerate(handles) if i in display],
                      [label for i, label in enumerate(labels) if i in display], loc='best')

            ax = fig.add_subplot(1, 2, 2)
            for wave in euc_valid_waveforms:
                if norm_method == 'min_max':
                    wave_n = preprocessing.minmax_scale(wave.values, feature_range=(0, 1), axis=0, copy=False)
                    ax.plot(utils.timeline(wave.values, fs), wave_n, color='lightblue', label='waveforms')
                elif norm_method == 'zscore':
                    wave_n = stats.zscore(wave.values)
                    ax.plot(utils.timeline(wave.values, fs), wave_n, color='lightblue', label='waveforms')
            ax.plot(mean_waveform_pd, color='royalblue', linewidth=2, label='Mean')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('PPG (a.u.)')
            ax.set_title("No similar signal removed - Original time reference")
            # Get artists and labels for legend and chose which ones to display
            handles, labels = ax.get_legend_handles_labels()
            display = (0, len(euc_valid_waveforms))
            ax.legend([handle for i, handle in enumerate(handles) if i in display],
                      [label for i, label in enumerate(labels) if i in display], loc='best')
            plt.show()

        plt.show()

    return euc_valid_waveforms, mean_waveform_pd
