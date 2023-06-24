from genre_classification.feature_extraction.feature_extraction import FeatureExtraction
from settings import sample_rate, hop_length


def get_feature_extraction() -> FeatureExtraction:
    """
     factory obj for feature extraction class
    """

    return FeatureExtraction(
        sample_rate=sample_rate,
        hop_length=hop_length
    )
