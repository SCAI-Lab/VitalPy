<h1 align="center">
<img src="https://github.com/SCAI-Lab/VitalPy/assets/33239037/e8a006a2-9f84-4fc4-abd0-e3d9ebfb37d0" width="500">
</h1><br>

# VitalPy: A Vital Signal Analysis Package

Currently, the package supports PPG preprocessing and extraction of more than 400 features. The PPG pipeline was originally implemented for analysis of the AuroraBP database.

**It provides:**
- PPG preprocessing: Singal quality metrics, baseline extraction, etc.
- PPG feature extraction: time-domain, frequency-domain, statistical features ( >400 features)
- Compatibility with PPG recorded from 128 Hz to 500 Hz: tested with local devices and large datasets.


## Use

VitalPy is written in Python (3.9+). Navigate to the Python repository and install the required packages:

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

<img src="https://github.com/SCAI-Lab/VitalPy/assets/33239037/9ede79d8-2613-47b6-a1ca-b5247c417cfc" width="650">

#### Preprocessing steps

All preprocessing steps are depicted. The final result should have filtered out all low quality waveforms.

<img src="https://github.com/SCAI-Lab/VitalPy/assets/33239037/ecc0a2ea-b4e8-4f67-a4b2-05232de1f81c" width="700">

#### PPG Keypoints

Exemplary PPG keypoint extraction.

<img src="https://github.com/SCAI-Lab/VitalPy/assets/33239037/93e17375-e5c3-44a3-a437-63aa9bbb2262" width="800">

## License

VitalPy is available under the General Public License v3.0.

## Citation

If you use this repository or any of its components and/or our paper as part of your research, please cite the publication as follows:
> A. Cisnal*et al.* "Robust Feature Selection for Continuous BP Estimation in Multiple Populations: Towards Cuffless Ambulatory BP Monitoring," *IEEE J Biomed Health Inform*, Under Review (2023).

```
@unpublished{vitalpy,
  title={Robust Feature Selection for Continuous BP Estimation in Multiple Populations: Towards Cuffless Ambulatory BP Monitoring},
  author={Cisnal, Ana and Li, Yanke and Fuchs, Bertram and Ejtehadi, Mehdi and Riener, Robert and Paez-Granados, Diego},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2023},
  note = "Under review"
}
```
