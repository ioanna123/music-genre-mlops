from typing import List, Any

from pydantic import BaseModel


class DatasetLoader(BaseModel):
    """
    Class representing an arbitrary segment.
    """
    train_dataloader: Any
    val_dataloader: Any
    classes: List[str]
    test_subset: Any
