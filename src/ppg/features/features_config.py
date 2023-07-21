from src.ppg.features.statistical_features import *
from src.ppg.features.temporal_features import *
from src.ppg.features.spectral_features import *

"""
Set the features to be calculated.
"""

# List of x% of pulse height of PPG for calculating branch width (BW) and branch width ratio (BWR)
X_VALUES_BW_FEATURES = [10, 25, 33, 50, 66, 75, 90]


TEMPORAL_FEATURES = [
    (autocorr, 'corr', True, 'corr'),
    (calc_centroid, 'centr', True, 'centroid'),
    (minpeaks, 'min_p', True, 'Min peaks'),
    (maxpeaks, 'max_p', True, 'Maxn peaks'),
    (mean_abs_diff, 'mean_abs_diff', True, 'Mean absolute differences'),
    (mean_diff, 'mean_diff', True, 'Mean difference'),
    (median_abs_diff, 'med_abs_diff', True, 'Median absolute differences'),
    (median_diff, 'med_diff', True, 'Median of differences'),
    (distance, 'dist', True, 'Distance'),
    (sum_abs_diff, 'sadif', True, 'Sum of absolute differences'),
    (zero_cross_1d, 'zc1d', True, 'Zero cross 1 derivative'),
    (zero_cross_2d, 'zc1d', True, 'Zero cross 2 derivative'),
    (zero_cross_3d, 'zc1d', True, 'Zero cross 3 derivative'),
    (total_energy, 'total_ene', True, 'Total energy'),
    (slope, 'slope', True, 'Slope'),
    (abs_energy, 'abs_ene', True, 'absolute energy'),
    (entropy_kde, 'ent_kde', True, 'Entropy KDE'),
    (entropy_gauss, 'ent_gauss', True, 'Entropy Gauss'),
]

STATISTIC_FEATURES = [
    (skewness, 's', True, 'skewness',),
    (kurtosis, 'k', True, 'kurtosis',),
    (mav, 'mav', True, 'mean absolute value'),
    (median, 'median', True, 'median',),
    (mad, 'mad', True, 'mean absolute deviation',),
    (med_ad, 'med_ad', True, 'median absolute deviation',),
    (rms, 'rms', True, 'rms',),
    (sd, 'sd', True, 'Standard Deviation',),
    (shape_factor, 'sf', True, 'Shape factor',),
    (impulse_factor, 'if', True, 'Impulse factor',),
    (crest_factor, 'cf', True, 'Crest factor',),
    (variance, 'v', True, 'Variance',),
    (irq, 'irq', True, 'Interquartile range',),
    (perfussion, 'p', True, 'perfussion',)
]

SPECTRAL_FEATURES = [
    (fsqi, 'fsqi', True),
    (fsqi_first, 'fsqi1', True),
    (fsqi_second, 'fsqi2', True),
    (fsqi_third, 'fsqi3', True),
    (frequency_first, 'f1', True),
    (mag_first, 'mag_f1', True),
    (frequency_second, 'f2', True),
    (mag_second, 'mag_f2', True),
    (frequency_third, 'f3', True),
    (mag_third, 'mag_f3', True),
    (spectral_distance, 'spectral_distance', True),
    (fundamental_frequency, 'fundamental_frequency', True,),
    (max_power_spectrum, 'max_power_spectrum', True,),
    (max_frequency, 'max_frequency', True,),
    (median_frequency, 'median_frequency', True,),
    (spectral_centroid, 'spectral_centroid', True,),
    (spectral_decrease, 'spectral_decrease', True,),
    (spectral_kurtosis, 'spectral_kurtosis', True,),
    (spectral_skewness, 'spectral_skewness', True,),
    (spectral_spread, 'spectral_spread', True,),
    (spectral_slope, 'spectral_slope', True,),
    (spectral_variation, 'spectral_variation', True,),
    (spectral_maxpeaks, 'spectral_maxpeaks', True,),
    (spectral_roll_off, 'spectral_roll_off', True,),
    (spectral_roll_on, 'spectral_roll_on', True,),
    (human_range_energy, 'human_range_energy', True,),
    (power_bandwidth, 'power_bandwidth', True,),
    (fft_mean_coeff, 'fft_mean_coeff', False,),
    (spectral_entropy, 'spectral_entropy', True,),
    (wavelet_entropy, 'wavelet_entropy', True,),
    (wavelet_abs_mean, 'wavelet_labs_mean', False,),
    (wavelet_std, 'wavelet_std', False,),
    (wavelet_var, 'wavelet_var', False,),
]
