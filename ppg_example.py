import pandas as pd
import os.path
from src.ppg.PPGSignal import PPGSignal

DATA_DIR = 'CHANGE'

# Load an example signal from the AuroraBP database
waveform_df = pd.read_csv(os.path.join(DATA_DIR, '/measurements_oscillometric/o000/o000.initial.Sitting_arm_down.tsv'), delimiter='\t')
waveform_df = waveform_df.rename(columns={'optical': 'ppg', 'ekg': 'ecg'})

# The dataframe should have the columns 't' for time and 'ppg' for the ppg signal values.
waveform_df = waveform_df[['t', 'ppg']]


# Check the signal (terminate after you have seen enough waveforms)
signal = PPGSignal(waveform_df, verbose=1)
signal.check_keypoints()

# Extract features
signal = PPGSignal(waveform_df, verbose=0)
features = signal.extract_features()
print(features)

