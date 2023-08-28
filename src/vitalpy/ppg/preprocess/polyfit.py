import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.vitalpy.ppg.params import POLY_COEFFS, POLY_COEFFS_DIAS, POLY_COEFFS_SYS
from src.vitalpy.ppg.utils import derivative


def template_polyfit(template: pd.Series, fs, n_coeff: int = POLY_COEFFS,
                     show: bool = False) -> pd.Series:
    """
    Calculate the approximated signal of the template using a polynomial fit.
    It can be useful for calculating the key-points.

    Parameters:
        template (pd.Series): Beat-to-beat PPG waveform or templates.
        fs (int): Sample frequency of the signal (Hz).
        n_coeff (int, optional): Coefficient of the polynomial approximation. Defaults to 15.
        show (bool, optional): Optional flag for showing the results. Defaults to False.

    Return:
        pd.Series: Polynomial approximation of the template.
    """

    time = np.linspace(0, len(template) / fs, len(template))
    coeffs = np.polyfit(time, template.to_numpy(), n_coeff)
    poly_template = np.poly1d(coeffs)(time)

    if show:
        plt.subplots(3, 1, figsize=(10, 4), sharex=True)

        plt.subplot(1, 3, 1)
        plt.title('PPG signal', fontsize='small')
        plt.plot(time, template.to_numpy(), label='raw')
        plt.plot(time, poly_template, ':', label='poly')
        plt.legend(bbox_to_anchor=(1.1, 1.05))

        plt.subplot(1, 3, 2)
        plt.title('First derivative of the PPG signal', fontsize='small')
        d1 = derivative(template.to_numpy(), fs)
        d1_fit = derivative(poly_template, fs)
        plt.plot(time[0:-1], d1, label='raw')
        plt.plot(time[0:-1], d1_fit, ':', label='poly')

        plt.subplot(1, 3, 3)
        plt.title('Second derivative of the PPG signal', fontsize='small')
        d2 = derivative(d1, fs)
        d2_fit = derivative(d1_fit, fs)
        plt.plot(time[0:-2], d2, label='raw')
        plt.plot(time[0:-2], d2_fit, ':', label='poly')
        plt.xlabel('Time (s)')
        plt.show()

    return pd.Series(data=poly_template, index=np.array(template.index))


def template_double_polyfit(template: pd.Series, fs, n_coeff_sys: int = POLY_COEFFS_SYS,
                            n_coeff_dias: int = POLY_COEFFS_DIAS, show: bool = False) -> pd.Series:
    """
    Calculate the approximated signal of the template using a polynomial fit.
    It can be useful for calculating the key-points.

    Parameters:
        template (pd.Series): Beat-to-beat single waveform or templates.
        fs (int): Sample frequency of the signal (Hz).
        n_coeff_sys (int, optional): Coefficient of the polynomial approximation for systolic part.
                                    Defaults to 5.
        n_coeff_dias (int, optional): Coefficient of the polynomial approximation for diastolic part.
                                     Defaults to 10.
        show (bool, optional): Optional flag for showing the results. Defaults to False.

    Return:
        pd.Series: Polynomial approximation of the template.
    """

    # Systolic peak is the maximum peak
    idx_s = template.idxmax()
    # split the template into systolic and diastolic part
    systolic_part = template.loc[template.index[0]:idx_s]
    diastolic_part = template.loc[idx_s:template.index[-1]]
    # fit each part
    poly_mean_sys = template_polyfit(systolic_part, fs, n_coeff_sys, False)
    poly_mean_dias = template_polyfit(diastolic_part, fs, n_coeff_dias, False)
    # put all together again
    poly_wave = pd.Series(poly_mean_sys.append(poly_mean_dias, ignore_index=False))

    if show:
        plt.subplots(3, 1, figsize=(10, 4), sharex=True)
        time = np.linspace(0, len(template) / fs, len(template))

        plt.subplot(1, 3, 1)
        plt.title('PPG signal', fontsize='small')
        plt.plot(template, label='raw')
        plt.plot(poly_wave, ':', label='poly')
        plt.legend(bbox_to_anchor=(1.1, 1.05))

        plt.subplot(1, 3, 2)
        plt.title('First derivative of the PPG signal', fontsize='small')
        d1 = derivative(template.to_numpy(), fs)
        d1_fit = derivative(poly_wave.to_numpy(), fs)
        plt.plot(time[0:-1], d1, label='raw')
        plt.plot(time[0:-1], d1_fit, ':', label='poly')

        plt.subplot(1, 3, 3)
        plt.title('Second derivative of the PPG signal', fontsize='small')
        d2 = derivative(d1, fs)
        d2_fit = derivative(d1_fit, fs)
        plt.plot(time[0:-2], d2, label='raw')
        plt.plot(time[0:-2], d2_fit, ':', label='poly')
        plt.xlabel('Time (s)')
        plt.show()

    return poly_wave
