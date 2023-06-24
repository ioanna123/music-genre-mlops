from genre_classification.preprocessor.audio_preprocess import AudioPreprocess
from genre_classification.preprocessor.image_dataset import ImageDataset
from settings import sample_rate


def get_audio_preprocessor() -> AudioPreprocess:
    return AudioPreprocess(sample_rate=sample_rate)


def get_dataset() -> ImageDataset:
    return ImageDataset()
