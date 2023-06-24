from typing import Any, Optional

from pydantic import BaseModel, confloat


class Segment(BaseModel):
    """
    Class representing an arbitrary segment.
    """
    start: confloat(ge=0.0)
    duration: confloat(ge=0.0)
    value: Optional[Any]
    label: Optional[str]

    @property
    def end(self):
        return self.start + self.duration
