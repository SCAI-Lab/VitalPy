from ..features import utils as feature
from ..preprocess.concatenate import clean_valid_dataset
from ..preprocess.polyfit import template_polyfit, template_double_polyfit


class FeatureExtractor:
    """
        Class for extracting various features from a waveform for plotting purposes.

        Args:
            wave (np.array): The input waveform (single PPG template).
            sampling_rate (int): The sampling rate of the waveform.
            key_points (dict): Keypoints of the wave, used for further feature calculations.

        Attributes:
            wave (np.array): The input waveform.
            sampling_rate (int): The sampling rate of the waveform.
            key_points (dict): Keypoints of the wave, used for further feature calculations.

        Methods:
            extract(): Extracts various features from the waveform.
            _extract_key_temporal(): Extracts temporal keypoints.
            _extract_temporal(): Extracts temporal features.
            _extract_bandwidth_features(): Extracts bandwidth features.
            _extract_polyfit_coefficients(): Extracts polyfit coefficient features.
            _extract_statistical_features(): Extracts statistical features.
            _extract_spectral_features(): Extracts spectral features.

        """
    def __init__(self, wave, sampling_rate, key_points):
        self.wave = wave
        self.sampling_rate = sampling_rate
        self.key_points = key_points

    def extract_features(self):
        """
        Extracts various features from the waveform.

        Returns:
            dict: A dictionary containing the extracted features for the specified waveform. Keys: Feature names; Values: Feature values
        """
        feature_fns = [
            self._extract_key_temporal,
            self._extract_temporal,
            self._extract_bandwidth_features,
            self._extract_polyfit_coefficients,
            self._extract_statistical,
            self._extract_spectral
        ]
        features = {}
        for fn in feature_fns:
            feats = fn()
            features.update(feats)
        return features

    def _extract_key_temporal(self):
        temp_key_feat = feature.extract_key_temporal(self.wave, self.sampling_rate, self.key_points, norm_method='z_score')
        return FeatureExtractor.append_to_keys(temp_key_feat, 'temp_key_')

    def _extract_temporal(self):
        temp_feat = feature.extract_temporal(self.wave, self.sampling_rate)
        return FeatureExtractor.append_to_keys(temp_feat, 'temp_')

    def _extract_bandwidth_features(self):
        bw_feat = feature.branch_width_features(self.wave, self.sampling_rate)
        return FeatureExtractor.append_to_keys(bw_feat, 'bw_')

    def _extract_polyfit_coefficients(self):
        coef_feat = feature.extract_coeffs(feature.normalize_template(self.wave, 'z_score'), self.sampling_rate)
        return FeatureExtractor.append_to_keys(coef_feat, 'poly_coeff_')

    def _extract_statistical(self):
        stat_feat = feature.extract_statistical(self.wave)
        return FeatureExtractor.append_to_keys(stat_feat, 'stat_')

    def _extract_spectral(self):
        spect_feat = feature.extract_spectral(self.wave, self.sampling_rate)
        return FeatureExtractor.append_to_keys(spect_feat, 'spect_')

    @staticmethod
    def append_to_keys(input_dict, appendix):
        output_dict = {}
        for key in input_dict.keys():
            output_dict[appendix + key] = input_dict[key]
        return output_dict


class PlottingFeatureExtractor:
    """
    Class for extracting various features from a waveform for plotting purposes.

    Args:
        wave (np.array): The input waveform (single PPG template).
        sampling_rate (int): The sampling rate of the waveform.
        show (bool): Whether to show plots during extraction.

    Attributes:
        wave (np.array): The input waveform.
        sampling_rate (int): The sampling rate of the waveform.
        show (bool): Whether to show intermediate results during extraction.

    Methods:
        extract(): Extracts various features from the waveform.
        _extract_raw_keypoints(): Extracts keypoints of the raw signal.
        _extract_poly1d_keypoints(): Extracts keypoints of the first order polyfit.
        _extract_poly2d_keypoints(): Extracts keypoints of the second order polyfit.
        _extract_bandwidth_features(): Extracts bandwidth features.
        _extract_statistical_features(): Extracts statistical features.
        _extract_temporal_features(): Extracts temporal features.
        extract_spectral_features(preprocessed_signal, raw_signal, sampling_rate): Extracts spectral features.

    """

    def __init__(self, wave, sampling_rate, show):
        self.wave = wave
        self.sampling_rate = sampling_rate
        self.show = show

    def extract_plotting_features(self):
        """
        Extracts various features from the waveform.

        Returns:
            dict: A dictionary containing the extracted features for the specified waveform. Keys: Feature names; Values: Feature values
        """
        feature_fns = [
            self._extract_raw_keypoints,
            self._extract_poly1d_keypoints,
            self._extract_poly2d_keypoints,
            self._extract_bandwidth_features,
            self._extract_statistical_features,
            self._extract_temporal_features
        ]
        features = {}
        for fn in feature_fns:
            feats = fn()
            features.update(feats)
        return features

    def _extract_raw_keypoints(self):
        # Keypoints of raw signal
        valid, key_points = feature.extract_key_points(self.wave, self.sampling_rate, show=self.show)
        return FeatureExtractor.append_to_keys(key_points, 'raw_kp_')

    def _extract_poly1d_keypoints(self):
        # Keypoints of first order polyfit
        poly_wave = template_polyfit(self.wave, self.sampling_rate, show=self.show)
        valid, key_points = feature.extract_key_points(poly_wave, self.sampling_rate, show=self.show)
        return FeatureExtractor.append_to_keys(key_points, 'poly1d_kp_')

    def _extract_poly2d_keypoints(self):
        # Keypoints of second order polyfit
        poly_wave = template_double_polyfit(self.wave, self.sampling_rate)
        valid, key_points = feature.extract_key_points(poly_wave, self.sampling_rate, show=self.show)
        poly2d_features = FeatureExtractor.append_to_keys(key_points, 'poly2d_kp_')
        if valid:
            feat = feature.extract_key_temporal(poly_wave, self.sampling_rate, key_points, norm_method='z_score')
            # features.update(FeatureExtractor.append_to_keys(feat, 'key_temp_'))
        return poly2d_features

    def _extract_bandwidth_features(self):
        # Bandwidth features
        feat = feature.branch_width_features(template=self.wave, fs=self.sampling_rate, x_values=[10, 33, 50, 66, 90])
        return FeatureExtractor.append_to_keys(feat, 'bw_')

    def _extract_statistical_features(self):
        # Statistical features
        feat = feature.extract_statistical(self.wave)
        return FeatureExtractor.append_to_keys(feat, 'stat_')

    def _extract_temporal_features(self):
        # Temporal features
        feat = feature.extract_temporal(self.wave, fs=self.sampling_rate)
        return FeatureExtractor.append_to_keys(feat, 'temp_')

    @staticmethod
    def extract_spectral_features(preprocessed_signal, raw_signal, sampling_rate):
        # Create a new dataframe only with the valid waves and maintaining the original time index
        valid_waveform_df = clean_valid_dataset(preprocessed_signal, raw_signal)

        # Spectral features
        feat = feature.extract_spectral(valid_waveform_df['f_ppg'], sampling_rate)
        return FeatureExtractor.append_to_keys(feat, 'spect_')

