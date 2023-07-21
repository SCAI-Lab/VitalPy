import pandas as pd
from src.ppg.preprocess.Preprocessor import Preprocessor
from src.ppg.params import PPG_SAMPLE_RATE
from src.ppg.features.FeatureExtractor import FeatureExtractor


class PPGSignal:
    """ Extract PPG features from a PPG signal given in pd.DataFrame format. The given signal will be preprocessed.
    Then beat-to-beat templates will be recognized and low SQI templates will be discarded.
    Finally, template-based and spectral-based features will be calculated and returned.

    !!!SQI (Signal Quality Index) filtering!!!: PPG sensors have varying recording properties. To guarantee the extraction of high
    quality templates, set the parameters in src/ppg/params.py such that they fit your signal.

    Args:
        signal (pandas.DataFrame): The DataFrame containing the PPG signal data in a column named
            signal['ppg'] and the time in signal[t] in UNIX format.
        verbose(int):   0: No plotting or logging output
                        1: Plotting and logging output
    """

    def __init__(self, signal, verbose=0):
        self.sampling_rate = PPG_SAMPLE_RATE
        self.show = False
        if verbose == 1:
            self.show = True

        # Load raw signal
        self.raw_signal = self._load_signal(signal)
        print('SIGNAL LOADED')

        # Preprocess raw signal
        self.preprocessed_signal, self.raw_signal = self._preprocess(self.raw_signal, self.sampling_rate)
        print('SIGNAL PREPROCESSED')

    @staticmethod
    def _load_signal(raw_signal):
        """Load signal from dataframe and check for signal compatibility.

        Args:
            raw_signal (pd.DataFrame): The DataFrame containing the raw signal data.

        Raises:
            ValueError: If the raw_signal DataFrame does not contain the required columns.

        Returns:
            pd.DataFrame: The loaded DataFrame representing the signal data.
        """

        required_cols = ['t', 'ppg']

        if any(col not in raw_signal.columns for col in required_cols):
            raise ValueError("Raw signal format incorrect - missing required columns: {}".format(required_cols))

        return raw_signal

    def _preprocess(self, raw_signal, sr):
        """Preprocess loaded signal.

        Args:
            raw_signal (pd.DataFrame): The DataFrame containing the raw signal data.
            sr (int): The sampling rate of the signal.

        Returns:
            preprocessed_signal: list of pd.Series, every pd.Series representing one processed PPG waveform
        """

        # Preprocess PPG signal
        # preprocessed_signal = go(waveform_df=raw_signal, fs=sr, show=self.show)
        preprocessor = Preprocessor(waveform_df=raw_signal, fs=sr, show=self.show)
        preprocessed_signal, waveform_df = preprocessor.process()

        return preprocessed_signal, waveform_df

    def extract_features(self):
        """Extract features from preprocessed signal. It includes keypoints, bandwidth , statistical, temporal and spectral features.

        Returns:
            dict: A dictionary containing the extracted features from the signal.
                  The dictionary includes both waveform-based features and spectral features.
        """

        wave_features_list = []
        for template in self.preprocessed_signal:
            # Extract features from single waveforms
            wave = FeatureExtractor(template, self.sampling_rate, self.show)
            features = wave.extract()
            wave_features_list.append(features)

        # Average features of all waveforms
        features_all_waves = pd.DataFrame(wave_features_list)
        feat_means = features_all_waves.mean().to_dict()

        # Extract spectral features from whole signal
        spectral_features = FeatureExtractor.extract_spectral_features(self.preprocessed_signal, self.raw_signal, self.sampling_rate)
        feat_means.update(spectral_features)

        return feat_means
