import pandas as pd


def clean_valid_dataset(gq_templates: list, waveform_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a dataframe containing only the data points of high quality PPG templates
    maintaining the index of the dataframe.

    Args:
        gq_templates (list): List of good quality templates (pd.Series)
        waveform_df (pd.DataFrame): Dataframe with signal data, such as ppg, f_ppg, fb_ppg, t, etc.

    Returns:
        pd.DataFrame: Dataframe with only data from those times at which the PPG signal has good quality.
    """

    # Crate a new dataframe which only contains the good data and maintains the original index (time)
    new_ppg = pd.Series(dtype=float)
    for d in gq_templates:
        new_ppg = pd.concat((new_ppg, d), axis=0)

    valid_waveform_df = pd.concat((new_ppg.rename('fb_ppg'), waveform_df.drop('fb_ppg', axis=1)), axis=1)
    valid_waveform_df.dropna(inplace=True)

    return valid_waveform_df
