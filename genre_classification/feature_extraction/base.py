from abc import ABC, abstractmethod
from typing import List

from genre_classification.data_model.segment import Segment


class FeatureExtractorBase(ABC):
    """
    Abstract class for an audio feature extraction components
    """

    @abstractmethod
    def transform(self, batch_of_wave_segments: List[Segment]) -> \
            List[Segment]:
        pass
