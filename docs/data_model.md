# Data Model and Schema Definition

### Introduction

This section explores the foundational components of the machine learning pipeline, focusing on data models and
enumerations. Data models play a crucial role in structuring and organizing data, while enumerations facilitate the
definition of criteria and options. Understanding the principles and usage of these constructs is essential for
developing an effective machine learning system.

### Data Model

Data models are instrumental in representing structured information and ensuring data integrity. Pydantic models, a
Python library, provide a structured approach to defining data schemas.

#### Segment Model

The Segment model represents an arbitrary segment and is used extensively in audio data processing. It defines
attributes such as start, duration, value, and label, facilitating the organization of segment-related information.

```python
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
```

#### Metadata Model

The Metadata model captures essential audio metadata, including format, streams, duration, sample rate, and channels.
This model is indispensable for understanding the characteristics of audio data.

```python
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
```

#### Evaluation Metrics Model

The EvaluationMetrics model quantifies the performance of machine learning models through metrics such as precision,
recall, and F1-score. This model aids in assessing classification and prediction tasks.

```python
from pydantic import BaseModel


class EvaluationMetrics(BaseModel):
    """
    Class representing the evaluation metrics.
    """
    precision: float
    recall: float
    fscore: float
```

#### Dataset Loader Model

The DatasetLoader model simplifies the management of datasets by encapsulating elements such as training dataloaders,
validation dataloaders, class labels, and test subsets.

```python
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
```

### Enumerations

Enumerations (enums) provide a structured approach to defining symbolic names bound to unique values. They enhance code
readability and prevent the use of incorrect or inconsistent values.

#### Criterion Enum

The Criterion enum offers a selection of loss functions, including cross-entropy, Kullback-Leibler divergence, and
smooth L1 loss. These loss functions are essential for training machine learning models.

```python
from enum import Enum

from torch import nn


class Criterion(Enum):
    """
    Class representing the criterion options.
    """
    cross_entropy = nn.CrossEntropyLoss()
    kldiv_loss = nn.KLDivLoss()
    smooth_loss = nn.SmoothL1Loss()
```

#### Transfer Learning Model Enum

The TLModel enum encompasses a range of pre-trained deep learning models, such as AlexNet, DenseNet-121, ResNet-18,
ResNet-34, and VGG. These pre-trained models serve as the foundation for audio genre classification tasks.

```python
from enum import Enum

class TLModel(Enum):
    """
    Class representing the criterion options.
    """
    alexnet = 'alexnet'
    densenet121 = 'densenet121'
    resnet18 = 'resnet18'
    resnet34 = 'resnet34'
    vgg = 'vgg'
```

### References

enum: https://docs.python.org/3/library/enum.html
data model: https://docs.pydantic.dev/latest/
