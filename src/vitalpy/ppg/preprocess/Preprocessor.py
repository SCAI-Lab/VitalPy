import matplotlib.pyplot as plt
from src.vitalpy.ppg.params import MIN_TEMPLATES, PPG_SAMPLE_RATE
from src.vitalpy.ppg.preprocess.filter_baseline import bandpass_filter, baseline_correction
from src.vitalpy.ppg.preprocess.segment import segmentation
from src.vitalpy.ppg.preprocess.similarity import euclidian_analysis
from src.vitalpy.ppg.preprocess.validate import validate_waveform


class Preprocessor:
    """
        Description:
        PPG signal preprocessing:
        1) Band pass filtering
        2) Baseline correction
        3) Segmentation
        4) Quality check stage 1 (morphology based)
        5) quality check stage 2 (similarity by euclidian distance)

         Parameters:
            waveform_df (pd.Dataframe): dataframe with column 't' (time) and 'ppg' (ppg distance)
            fs: sample frequency of the signal (Hz)
            show (bool): True/False for showing graphical results
        Return:
            clean data (list of pd.Series): list with all good quality beat-to-beat waveforms
                                            they are pd.Series so its index is the time
           """

    def __init__(self, waveform_df, fs=PPG_SAMPLE_RATE, show: bool = False):
        self.waveform_df = waveform_df
        self.fs = fs
        self.filtered_df = None
        self.corrected_df = None
        self.segmented_data = None
        self.validated_data = None
        self.clean_data = []
        self.show = show

    def filter(self):
        self.filtered_df = self.waveform_df.copy()
        self.filtered_df['f_ppg'] = bandpass_filter(self.waveform_df.ppg.to_numpy(), 0.25, 10, self.fs)

    def correct_baseline(self):
        self.corrected_df = self.filtered_df.copy()
        self.corrected_df['fb_ppg'] = baseline_correction(self.filtered_df['f_ppg'].to_numpy())

    def segment(self):
        self.segmented_data = segmentation(self.corrected_df['fb_ppg'])

    def validate(self):
        self.validated_data = validate_waveform(self.segmented_data, self.fs)

    def check_similarity(self):
        if len(self.validated_data) >= MIN_TEMPLATES:
            clean_data, _ = euclidian_analysis(self.validated_data, self.fs, show=self.show)
            if len(clean_data) >= MIN_TEMPLATES:
                self.clean_data = clean_data
        return

    def process(self):
        self.filter()
        self.correct_baseline()
        self.segment()
        self.validate()
        self.check_similarity()
        if self.show:
            self.plot()
        return self.clean_data, self.corrected_df

    def plot(self):
        fig, axes = plt.subplots(nrows=6, ncols=1, sharex=False, sharey=False, figsize=(9, 6))
        fig.tight_layout()
        fig.suptitle('PPG Signal Preprocessing', y=1.05, fontsize='large', weight='bold')

        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        xlim = [self.corrected_df.t.to_numpy()[0], self.corrected_df.t.to_numpy()[-1]]

        ax1.set_title(' Raw signal', fontsize='medium')
        ax1.plot(self.corrected_df.t, self.corrected_df['ppg'], 'k-', label='Raw')
        ax1.set_ylabel('PPG (a.u.)')

        ax2.set_title('Band-pass filtered', fontsize='medium')
        ax2.plot(self.corrected_df.t, self.corrected_df['f_ppg'], 'g-', label='Band-pass filtered')
        ax2.set_ylabel('PPG (a.u.)')

        ax3.set_title('Baseline removal', fontsize='medium')
        ax3.plot(self.corrected_df.t, self.corrected_df['fb_ppg'], 'c-', label='Baseline removed')
        ax3.set_ylabel('PPG (a.u.)')

        ax4.set_title('Extracted beat-to-beat waveforms - Phase 1', fontsize='medium')
        for d in self.segmented_data:
            ax4.plot(d.index / self.fs, d)
        ax4.set_ylabel('PPG (a.u.)')

        ax5.set_title('Extracted beat-to-beat waveforms - Phase 2', fontsize='medium')
        for d in self.validated_data:
            ax5.plot(d.index / self.fs, d)
        ax5.set_ylabel('PPG (a.u.)')

        ax6.set_title('Similarity checking', fontsize='medium')
        for d in self.clean_data:
            ax6.plot(d.index / self.fs, d)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('PPG (a.u.)')

        for ax in axes.flatten()[0:5]:
            ax.spines['top'].set(visible=False)
            ax.spines['right'].set(visible=False)
            ax.spines['bottom'].set(visible=False)
            ax.axes.get_xaxis().set_ticks([])
            ax.set_xlim(xlim)

        ax6.spines['right'].set(visible=False)
        ax6.spines['top'].set(visible=False)
        ax6.spines['top'].set(visible=False)
        ax6.set_xlim(xlim)
        fig.align_ylabels(axes.flatten())
        plt.show()
        return
