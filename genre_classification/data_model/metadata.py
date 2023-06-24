from typing import Any, List

from pydantic import BaseModel


class Metadata(BaseModel):
    """
    Class representing the audio metadata.
    """
    format: Any
    streams: List[Any]

    @property
    def duration(self) -> float:
        return float(self.streams[0]['duration'])

    @property
    def sample_rate(self) -> int:
        return int(self.streams[0]['sample_rate'])

    @property
    def channels(self) -> int:
        return int(self.streams[0]['channels'])
