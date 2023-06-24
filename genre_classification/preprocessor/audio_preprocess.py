import os
from subprocess import call
from typing import Generator, List

from librosa import load as librosa_load

from genre_classification.data_model.segment import Segment
from genre_classification.utils.metadata import extract_audio_metadata


class AudioPreprocess:
    def __init__(
            self,
            sample_rate: int

    ):

        self.sr = sample_rate

    def _audio_converter(self, source_audio_path: str, audio_format: str = 'wav'):
        if source_audio_path.endswith(audio_format):
            return source_audio_path
        else:
            audio_file_wav = os.path.join(source_audio_path.split('.')[:-1][0], audio_format)
            command = f"ffmpeg -i {source_audio_path} -acodec pcm_s16le -ac 1 -ar {self.sr} {audio_file_wav} -hide_banner -y"
            call(command, shell=True)
            return audio_file_wav

    def stream(self, source_audio: str, start: float, window_duration: float) -> Generator[
        List[Segment], None, None]:
        audio_file = self._audio_converter(source_audio_path=source_audio)
        meta = extract_audio_metadata(audio_file)
        if start + window_duration > meta.duration:
            window_duration = meta.duration - start
        waveform, sr = librosa_load(
            audio_file,
            offset=start,
            duration=window_duration,
            sr=self.sr,  # if sr is None then the actual sampling rate is returned
        )

        yield Segment(
            start=start,
            duration=window_duration,
            value=waveform
        )
