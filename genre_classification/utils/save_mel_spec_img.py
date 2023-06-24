import os

import librosa
import librosa.display
import matplotlib.pyplot as plt

from genre_classification.data_model.segment import Segment
from settings import sample_rate, hop_length


def save_mel_spec_per_genre(image_dir: str, genre: str, image_name: str, mel_spec: Segment):
    os.makedirs(os.path.join(image_dir, genre), exist_ok=True)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    img = librosa.display.specshow(mel_spec.value, sr=sample_rate, hop_length=hop_length, cmap='cool', ax=ax)
    fig.savefig(os.path.join(image_dir, genre, f'{image_name}_{mel_spec.start}_{mel_spec.duration}.png'))
    plt.close()
