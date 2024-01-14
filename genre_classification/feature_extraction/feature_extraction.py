import librosa
import numpy as np

from genre_classification.data_model.segment import Segment
from genre_classification.feature_extraction.base import FeatureExtractorBase


class FeatureExtraction(FeatureExtractorBase):
    """
    Feature Extractor class from audio segments.
    Transforming the Audio Files into Mel Spectrograms
    """

    def __init__(
            self,
            sample_rate: int,
            hop_length: int,

    ):
        self.sr = sample_rate
        self.hop_length = hop_length

    def get_melspectrogram(self, audio_data):
        """Return the melspectrogram of the audio."""

        return librosa.feature.melspectrogram(y=audio_data.ravel(), sr=self.sr, hop_length=self.hop_length)

    @staticmethod
    def amplitute_to_db(mel_spec):
        """Return the amplitude_to_db mel spectrogram"""

        return librosa.amplitude_to_db(mel_spec, ref=np.max)

    def transform(
            self,
            wave_segment: Segment
    ) -> Segment:
        mel_spec = self.get_melspectrogram(audio_data=wave_segment.value)
        mel_spec_db = self.amplitute_to_db(mel_spec=mel_spec)
        return Segment(
            start=wave_segment.start,
            duration=wave_segment.duration,
            actual_duration=wave_segment.duration,
            value=mel_spec_db)
