# VitalPy: A Vital Signal Analysis Package

Currently, the package supports PPG preprocessing and extraction of more than 400 features. The PPG pipeline was originally implemented for analysis of the AuroraBP database.

**It provides:**
- PPG preprocessing
- PPG feature extraction ( >400 features)

## Use

Navigate to the Python repository and install the required packages:

```pip install -r requirements.txt```

Import:

```from src.ppg.PPGSignal import PPGSignal```

Check the signal (make sure that the file waveform_df is in dataframe format and contains the columns 't' for time and 'ppg' for the signal values):

```signal = PPGSignal(waveform_df, verbose=1)```

```signal.check_keypoints()```

Get features:

```signal = PPGSignal(waveform_df, verbose=0)```

```features = signal.extract_features()```



## Example plots for a AuroraBP signal

Used file: measurements_oscillometric/o001/o001.initial.Sitting_arm_down.tsv

#### Mean signal

The following figure shows the mean template computed from all templates within the signal given as input.

<img src="https://github.com/SCAI-Lab/VitalPy/assets/33239037/5a8136b0-f9fc-49a4-b345-d2609a5113de" width="150">
#### Preprocessing steps

All preprocessing steps are depicted. The final result should have filtered out all low quality waveforms.

<img src="https://github.com/SCAI-Lab/VitalPy/assets/33239037/f9e518d6-4fa7-4a7b-b9f8-990c20b87a51" width="150">

#### PPG Keypoints

Exemplary PPG keypoint extraction.

<img src="https://github.com/SCAI-Lab/VitalPy/assets/33239037/84c44e52-2a07-4faf-b142-45a8e17486ee" width="150">

## License

VitalPy is available under the General Public License v3.0.

## Citation

Robust Feature Selection for Continuous BP Estimation in Multiple Populations: Towards Cuffless Ambulatory BP Monitoring

Ana Cisnal, Yanke Li, Bertram Fuchs, Mehdi Ejtehadi, Robert Riener, and Diego Paez-Granados
