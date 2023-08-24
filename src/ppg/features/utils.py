import pandas as pd
from scipy import integrate, stats
from scipy.signal import argrelmax, argrelmin
import src.ppg.utils as utils
from src.ppg.params import PPG_SAMPLE_RATE, POLY_COEFFS
import numpy as np
from sklearn import preprocessing

# default file for setting which features to calculate
from src.ppg.features.features_config import STATISTIC_FEATURES, TEMPORAL_FEATURES, SPECTRAL_FEATURES, X_VALUES_BW_FEATURES


def extract_coeffs(single_wave: np.ndarray, fs: int = PPG_SAMPLE_RATE) -> dict:
    time = utils.timeline(single_wave, fs)
    feat_dict = {}
    coeffs = np.polyfit(time, single_wave, POLY_COEFFS)
    for c in range(len(coeffs)):
        feat_dict[f'coef{c}'] = coeffs[c]

    return feat_dict


def extract_spectral(waveform: pd.Series, fs: int = PPG_SAMPLE_RATE) -> dict:
    single_wave = waveform.to_numpy()  # it can be template or whole measurement
    feat_dict = {}

    for function, name, use in SPECTRAL_FEATURES:
        if use:
            if utils.n_arg_fun(function) == 1:
                feat_dict[f'{name}'] = function(single_wave)
            else:
                feat_dict[f'{name}'] = function(single_wave, fs)

    return feat_dict


def extract_statistical(waveform: pd.Series) -> dict:
    single_wave = waveform.to_numpy()
    feat_dict = {}

    for function, name, use, abr in STATISTIC_FEATURES:
        if use:
            feat_dict[name] = function(single_wave)

    return feat_dict


def extract_temporal(waveform: pd.Series, fs: int = PPG_SAMPLE_RATE) -> dict:
    single_wave = waveform.to_numpy()
    feat_dict = {}

    for function, name, use, abr in TEMPORAL_FEATURES:
        if utils.n_arg_fun(function) == 1:
            feat_dict[name] = function(single_wave)
        else:
            feat_dict[name] = function(single_wave, fs)

    return feat_dict


def normalize_template(template: pd.Series, method: str = 'z_score'):
    """
    Description:
        Create a dataframe containing only the data points of good quality PPG templates
        maintaining the index of the dataframe.

    Parameters:
            template (pd.Series): PPG template
            method (str): normalization method (z_score, min_max, none)
    Return:
            n_template: normalized template
    """

    if method == 'z_score':
        n_template = stats.stats.zscore(template)
        n_template = n_template.to_numpy()
    elif method == 'min_max':
        n_template = preprocessing.minmax_scale(template, feature_range=(0, 1), axis=0, copy=False)
    elif method == 'none':
        n_template = template.to_numpy()
    else:
        n_template = template.to_numpy()
        print("Warning. Normalization method has not been appropriately selected in 'normalize_template' def")

    return n_template


def extract_key_temporal(waveform: pd.Series, fs: float, idx_points: dict,
                         norm_method: str = 'z_score') -> dict:
    # Normalized signal
    single_wave = normalize_template(template=waveform, method=norm_method)
    # Create dictionary for storing temporal features
    feat_dict = {}

    v_points = {}  # Create a dictionary with the values (y-values) of each index
    all_names = []  # Create a list with all the names of the key points
    for k, v in idx_points.items():
        v_points[k] = single_wave[v]
        all_names.append(k)  # all_names = ['o', 'a', 'md', 'b','s','dn','ip','d', 'v']

    # INTENSITY OR AMPLITUDE RELATED FEATURES
    # mean intensity
    feat_dict['Im'] = np.mean(waveform)
    # mean intensity ratio if break the waveform in two parts by point x1
    for x1 in all_names[1:8]:  # except v and v1
        feat_dict[f'mean_ir{x1}'] = np.mean(single_wave[idx_points[x1]:]) / np.mean(single_wave[:idx_points[x1]])

    # All possible combinations.
    for idx1, x1 in enumerate(all_names):
        # Absolute intensity
        feat_dict[f'i_{x1}'] = v_points[x1]
        for x2 in all_names[idx1 + 1:]:
            # Intensity from x1 and x2 (Relative intensity)
            feat_dict[f'i_{x1}_{x2}'] = v_points[x1] - v_points[x2]
            # Intensity ratio Ix1/Ix2
            feat_dict[f'ir_{x1}_{x2}'] = v_points[x1] / v_points[x2]

    # Normalized intensity (0-1, being 0 'o' and 1 's')
    for x1 in all_names[1:]:  # all names except 'v' ('s' is always one but is later used)
        feat_dict[f'ni_{x1}'] = (v_points[x1] - v_points['o']) / (v_points['s'] - v_points['o'])

    # normalized between points
    comb = all_names[1:]  # except 'o' since it is zero
    for idx1, x1 in enumerate(comb):
        for x2 in comb[idx1 + 1:]:
            feat_dict[f'ni_{x1}_{x2}'] = feat_dict[f'ni_{x1}'] - feat_dict[f'ni_{x2}']

    # TIME RELATED FEATURES
    # Absolute values
    feat_dict['t_it'] = get_it_time(waveform, fs)
    for x1 in all_names[1:]:  # except 'o' (always 0)
        feat_dict[f't_{x1}'] = idx_points[x1] / fs
        # includes: t_v1 = length pulse

    for x1 in all_names[1:8]:  # except 'o' and 'v' (always 0 and 1)
        # Normalized times (t_x1/t_v1)
        feat_dict[f'nt_{x1}'] = idx_points[x1] / idx_points['v1']

    comb = all_names[1:8]
    for idx1, x1 in enumerate(comb):  # except 'o' and 'v' (always 0 and 1)
        for x2 in comb[idx1 + 1:]:
            # Time ratio between x1 and x2  TR = tx1/tx2
            feat_dict[f'tr_{x1}_{x2}'] = feat_dict[f't_{x1}'] / feat_dict[f't_{x2}']

    # All possible combinations
    comb = all_names[1:]
    for idx1, x1 in enumerate(comb):  # except 'o'
        for x2 in comb[idx1 + 1:]:
            # Time between x1 and x2 (t_x1_x2), which includes some
            # interesting times such as diastolic time (DT) = tv-ts or tsd
            feat_dict[f't_{x1}_{x2}'] = feat_dict[f't_{x1}'] - feat_dict[f't_{x2}']
            # Normalized time ratio between x1 and x2 and total length. TR = (tx1-tx2)/tt
            feat_dict[f'trn_{x1}_{x2}'] = feat_dict[f't_{x1}_{x2}'] / feat_dict['t_v1']

    # SLOPE
    # Includes ascending slope (AS) (s-o)/(ts-to) and to=0
    for x1 in all_names[1:]:  # all names except 'v'
        feat_dict[f'slp_{x1}'] = (v_points[x1] - v_points['o']) / feat_dict[f't_{x1}']
        # feat_dict[f'n_slp_{x1}'] = feat_dict[f'ni_{x1}'] / feat_dict[f't_{x1}']
        if x1 != 'v1':  # always 1
            # feat_dict[f'nn_slp_{x1}'] = feat_dict[f'ni_{x1}'] / feat_dict[f'nt_{x1}']
            feat_dict[f'slp_{x1}_n'] = (v_points[x1] - v_points['o']) / feat_dict[f'nt_{x1}']

    # DERIVATIVES-RELATED FEATURES
    derivative1 = utils.derivative(waveform=single_wave, fs=fs)
    derivative2 = utils.derivative(waveform=derivative1, fs=fs)

    # first derivative
    comb = ['md', 'a', 'b', 'dn', 'ip']  # 'o' and 's' are zero
    for name in comb:
        feat_dict[f'id_{name}'] = derivative1[idx_points[name]]

    for idx1, x1 in enumerate(comb):
        for x2 in comb[idx1 + 1:]:
            feat_dict[f'ird_{x1}_{x2}'] = feat_dict[f'id_{x1}'] / feat_dict[f'id_{x2}']

    # second derivative
    comb = ['o', 'a', 'b', 's', 'dn', 'ip', 'd']  # md is zero
    for name in comb:
        feat_dict[f'id2_{name}'] = derivative2[idx_points[name]]
    for idx1, x1 in enumerate(comb):
        for x2 in comb[idx1 + 1:]:
            feat_dict[f'ird2_{x1}_{x2}'] = feat_dict[f'id2_{x1}'] / feat_dict[f'id2_{x2}']

    # AREA BASED FEATURES
    try:
        period = 1 / fs
        total_area = integrate.simps(single_wave[idx_points['o']:idx_points['v']], dx=period)
        # All possible combinations of areas and ratios between total area
        comb = ['o', 'a', 'md', 'b', 's', 'dn', 'd', 'v']  # except ip
        for idx1, x1 in enumerate(comb):
            for x2 in comb[idx1 + 1:]:
                # Area from x1 and x2
                feat_dict[f'a_{x1}_{x2}'] = integrate.simps(single_wave[idx_points[x1]:idx_points[x2]], dx=period)
                # Area ratio of A1 (area from x1 and x2) and total_area
                feat_dict[f'art_{x1}_{x2}'] = feat_dict[f'a_{x1}_{x2}'] / total_area
        # Area ratio using one point
        for name in all_names[2:8]:  # all except 'o', 'a' and 'v'
            feat_dict[f'ar_{name}'] = ratio_areas(single_wave, period, idx_points[name])
    except:
        ("print error in area using %s %s %d %d" % (x1, x2, idx_points[x1], idx_points[x2]))

    # OTHER FEATURES
    # #Reflexion index or augmentation index (RI or AI)
    feat_dict['RI'] = v_points['ip'] / v_points['s']  # o i_ip /i_s
    # Large artery stiffness index
    # H = 1 -> height of the subject but can be approximated to 1
    feat_dict['LASI'] = 1 / (feat_dict['t_ip'] - feat_dict['t_s'])
    # #Normalized pulse volume
    feat_dict['mNPV'] = (v_points['s'] - v_points['o']) / feat_dict['Im']
    # PPG characteristic value or K value
    feat_dict['PPGK'] = (feat_dict['Im'] - v_points['o']) / (v_points['s'] - v_points['o'])
    return feat_dict


def branch_width_features(template: pd.Series, fs: float,
                          x_values: list = X_VALUES_BW_FEATURES) -> dict:
    """""
    Description:
        Calculate branch width (BW) features from a PPG template

    Parameters:
        template (pd.Series): PPG template
        fs (float): sample frequency of the signal (Hz)
        x_values (list): list of x value for calculating BW features

    Return:
        bw_features (dict): set of BW features (SBW, DBW, BW and BWR) for the defined x values
    """

    bw_features = {}
    # Systolic peak is the maximum peak
    idx_s = np.argmax(template.to_numpy())

    for x in x_values:
        # Systolic branch width (SBWx) at x% of pulse height of PPG
        bw_features[f'sbw{x}'] = get_sbw(template, idx_s, fs, x / 100)
        # Diastolic branch width (DBWx) at x% of pulse height of PPG
        bw_features[f'dbw{x}'] = get_dbw(template, idx_s, fs, x / 100)
        # Branch width (BWx) at x% of the pulse height of PPG (SBWx + DBWx)
        bw_features[f'bw{x}'] = bw_features[f'sbw{x}'] + bw_features[f'dbw{x}']
        # Branch width ratio (BWRx) at x % of the pulse height of PPG (DBWx/SBWx)
        bw_features[f'bwr{x}'] = bw_features[f'sbw{x}'] / bw_features[f'dbw{x}']
    return bw_features


def extract_key_points(template: pd.Series, fs: float, show: bool = False):
    """""
    Description:
        Extraction of the key-points (fiducial points) of the PPG signal template using derivative analysis.

    Parameters:
        template (pd.Series): PPG waveform
        fs: sample frequency of the signal (Hz)
        show (bool): show graphic

    Return:
        valid (bool): whether the waveform is valid and thus, the key-points has been successfully determined
        idx_points (dict):  key_point indexes of the waveform template
    """

    # Normalize waveform
    waveform = stats.stats.zscore(template)

    # Onset point 'o' is the first point
    idx_o = 0

    # Systolic peak 's' is the maximum peak
    idx_s = np.argmax(waveform)

    # Split the waveform in two parts: systolic and diastolic
    systolic_part = waveform[0:idx_s]
    diastolic_part = waveform[idx_s:len(waveform)]

    # Calculate first and second derivative
    sys_derivative1 = utils.derivative(systolic_part, fs)
    dias_derivative1 = utils.derivative(diastolic_part, fs)

    sys_derivative2 = utils.derivative(sys_derivative1, fs)
    dias_derivative2 = utils.derivative(dias_derivative1, fs)

    # Calculate MD (maximum derivative), also known as MS (maximum slope): maximum point of the first derivative
    idx_rel_max_ms = argrelmax(sys_derivative1)[0]  # get relative maximum (relative peaks)
    idx_max_ms = np.argmax(sys_derivative1[idx_rel_max_ms])  # get the maximum peaks
    idx_ms = idx_rel_max_ms[idx_max_ms]

    # Maximum peak at the systolic part of the second derivative ('a' or 'md2' point)
    # maximum second derivative (ta, a)
    try:
        idx_rel_max_md2 = argrelmax(sys_derivative2[0:idx_ms])[0]  # get relative maximum (relative peaks)
        idx_max_md2 = np.argmax(sys_derivative2[0:idx_ms][idx_rel_max_md2])  # get the maximum peaks
        idx_md2 = idx_rel_max_md2[idx_max_md2]
    except:  # if there is not a relative maximum... get the idx of the maximum value
        idx_md2 = np.argmax(sys_derivative2[1:idx_ms]) + 1

    mwin = int(fs * 0.04)
    # Minimum peak at the systolic part of the second derivative ('b point')
    idx_b = np.argmin(sys_derivative2[idx_md2:idx_s - mwin]) + idx_md2

    try:
        # Diastolic peak: it is the maximum peak after the Systolic peak in a window of 100-300 ms
        min_window = int(fs * 0.08)
        window_len = int((len(waveform) - idx_s) * 0.6)  # 60% of the duration of diastolic part
        dn_window = int(fs * 0.01)
        idx_rel_max = argrelmax(diastolic_part[min_window:window_len])[0]

        # there is no clear diastolic peak (no peak means no cross-zero in first derivative)
        if len(idx_rel_max) == 0:
            # get the maximum peak of the first derivative
            idx_rel_max_d1 = argrelmax(dias_derivative1[min_window:window_len])[0]
            idx_max_d1 = np.argmax(dias_derivative1[min_window:window_len][idx_rel_max_d1])
            _idx_d = idx_rel_max_d1[idx_max_d1] + min_window
            _idx_ip = _idx_d

            # dn is the maximum point of the second derivative
            idx_rel_max_d2 = argrelmax(dias_derivative2[0:_idx_d - dn_window])[0]
            _idx_dn = idx_rel_max_d2[-1]  # get the closest one to D

        # there is one clear diastolic peak
        elif len(idx_rel_max) == 1:
            _idx_d = int(idx_rel_max + min_window)

            # IP is the local maximum of the first derivative (before D)
            idx_rel_max_d1 = argrelmax(dias_derivative1[min_window:_idx_d])[0]
            _idx_ip = idx_rel_max_d1[-1] + min_window  # get the closest to D if there is more than one

            # DN is the local minimum of ppg (before ip) - first derivative cross - to +
            idx_rel_min_d1 = argrelmin(diastolic_part[min_window:_idx_ip])[0]
            _idx_dn = idx_rel_min_d1[-1] + min_window

        # there is more than one clear diastolic peak
        else:
            # get ip - get the maximum peak of the first derivative
            idx_rel_max_d1 = argrelmax(dias_derivative1[min_window:window_len])[0]
            idx_max_d1 = np.argmax(dias_derivative1[min_window:window_len][idx_rel_max_d1])
            _idx_ip = idx_rel_max_d1[idx_max_d1] + min_window

            # D: local maximum right after IP
            idx_rel_max_d1d = argrelmax(diastolic_part[_idx_ip:window_len])[0]
            _idx_d = idx_rel_max_d1d[0] + _idx_ip

            # DN is the local minimum of ppg (right before ip)
            # first derivative cross - to +        #same as there a clear diastolic point
            idx_rel_min_d1 = argrelmin(diastolic_part[min_window:_idx_ip])[0]
            _idx_dn = idx_rel_min_d1[-1] + min_window

        idx_d = _idx_d + idx_s
        idx_dn = _idx_dn + idx_s
        idx_ip = _idx_ip + idx_s
        valid = True

    except:
        # These key-points are not valid. Define as idx_1 - 3 only for plotting purposes
        idx_d = len(waveform) - 3
        idx_dn = len(waveform) - 3
        idx_ip = len(waveform) - 3
        valid = False

    idx_points = {'o': idx_o,               # Onset point 'o'
                  'a': idx_md2,             # 'a' point or maximum point of second derivative 'md2'
                  'md': idx_ms,             # maximum derivative 'md' or maximum slope 'ms'
                  'b': idx_b,               # 'b' point
                  's': idx_s,               # Systolic 's' peak
                  'dn': idx_dn,             # Dicrotic notch 'dn'
                  'ip': idx_ip,             # Inflection point 'ip'
                  'd': idx_d,               # Diastolic 'd' peak
                  'v': len(waveform) - 1}   # Valley point 'v'

    if show:
        utils.plot_key_points_dev(template, idx_points, fs)

    return valid, idx_points  # valid (True or False) and key-point indexes


def ratio_areas(template: np.ndarray, period: float, point: int) -> float:
    """""
    Description:
        Break the PPG curve into two parts based on a predefined key-point.
        Get the ratio of the area under the PPG curve of the first part (A1) and the second part (A2).
        RA = A1/A2
        Areas under the curve (A1 and A2) are calculated according to Simpson's rule

    Parameters:
        template (np.array): PPG template
        period: sample period of the signal (s)
        point: index of the predefined key-point for breaking the PPG template in two parts

    Return:
        (float) ratio area
    """
    a1 = integrate.simps(template[point:-1], dx=period)
    a2 = integrate.simps(template[0:point], dx=period)
    return a1 / a2


def get_sbw(template: pd.Series, idx_s: int, fs: float, x: float) -> float:
    """""
    Description:
        Systolic branch width at x% of pulse height of PPG

    Parameters:
        template (pd.Series): PPG template
        idx_s (int): index of the systolic peak (S) of the template
                    it can be calculated calling:
                    key_points = extract_key_points (template, fs)
                    idx_s = key_points['s']
        fs: sample frequency of the signal (Hz)
        x: x value

    Return:
        (float) Systolic branch width
    """

    template_np = template.to_numpy()
    s = template_np[idx_s]
    o = template_np[0]
    sbwx = idx_s - np.argmin(abs(template_np[0:idx_s] - (x * (s - o) + o)))
    return float(sbwx) / fs


def get_dbw(template: pd.Series, idx_s: int, fs: float, x: float) -> float:
    """""
    Description:
        Systolic branch width at x% of pulse height of PPG

    Parameters:
        template (pd.Series): PPG template
        idx_s (int): index of the systolic peak (S) of the template
                    it can be calculated calling:
                    key_points = extract_key_points (template, fs)
                    idx_s = key_points['s']
        fs: sample frequency of the signal (Hz)
        x: x value

    Return:
        Systolic branch width
    """
    template_np = template.to_numpy()
    s = template_np[idx_s]
    o = template_np[0]
    dbwx = np.argmin(abs(template_np[idx_s:-1] - (x * (s - o) + o)))
    return float(dbwx) / fs


def get_it_time(template: pd.Series, fs: int):
    """""
    Description:
        Get  intersecting-tanget point (IT) from the waveform, which is the
        intersecting point between the tangent lines of the maximum derivative at the systolic part
        and the onset point of the waveform

    Parameters:
        template (pd.Series): PPG template
        fs: sample frequency of the signal (Hz)

    Return:
        Intersection point relative time
    """
    waveform = template.to_numpy()

    # Systolic peak is the maximum peak
    t_s = np.argmax(template)

    # Split the waveform in two parts
    systolic_part = waveform[0:t_s]

    # Calculate first derivative
    sys_derivative1 = utils.derivative(systolic_part, fs)

    # MS (maximum slope) key point: maximum point of the first derivative
    t_ms = np.argmax(sys_derivative1)

    m_tg_ms = sys_derivative1[t_ms]  # slope of tg line at MS
    b_tg_ms = -m_tg_ms * t_ms / fs + waveform[t_ms]  # b= -m*x+y
    m_tg_o = 0
    b_tg_o = waveform[0]

    t_it, it = utils.get_intersection_point(m_tg_ms, b_tg_ms, m_tg_o, b_tg_o)

    return t_it
