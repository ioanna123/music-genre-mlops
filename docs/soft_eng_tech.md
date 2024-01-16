# Software Engineering Techniques in the Machine Learning Pipeline

## Introduction

This section delves into the software engineering techniques applied throughout the machine learning pipeline. Each step
of the pipeline, including data preprocessing, feature extraction, and training, adheres to a structured design pattern.
The adoption of these techniques enhances code modularity, maintainability, and scalability.

## Factory Pattern

The Factory Pattern is a creational design pattern that provides an interface for creating objects but allows subclasses
to alter the type of objects that will be created. In the context of the machine learning pipeline, the Factory Pattern
plays a pivotal role in creating instances of each pipeline step dynamically.

### Why Factory Pattern

The Factory Pattern is useful in the machine learning pipeline for the following reasons:

* Abstraction: It abstracts the process of object creation, encapsulating the details within factory methods.
* Flexibility: By using factory methods, the pipeline can create different feature extraction components without
  explicitly specifying their classes, enhancing flexibility.
* Dependency Injection: Factory methods facilitate dependency injection by allowing the pipeline to inject the
  appropriate feature extraction component, promoting modularity and testability.

### Implementation of feature extraction class

* Abstract Base Class (ABC) for Feature Extraction
  An abstract base class, FeatureExtractorBase, defines the common methods that feature extraction components must
  implement. This abstraction enforces a consistent interface for all feature extractors.

  ```python
  # Abstract Base Class (ABC) for Feature Extraction
  from abc import ABC, abstractmethod
  from typing import List
  
  from genre_classification.data_model.segment import Segment
  
  class FeatureExtractorBase(ABC):
      @abstractmethod
      def transform(self, batch_of_wave_segments: List[Segment]) -> List[Segment]:
          pass
  ```

* Implementation Class for Feature Extraction
  The FeatureExtraction class implements the transform method, which includes all the necessary code for audio feature
  extraction. It inherits from the abstract base class, ensuring adherence to the common interface.

  ```python
  # Implementation Class for Feature Extraction
  import librosa
  import numpy as np
  
  from genre_classification.data_model.segment import Segment
  from genre_classification.feature_extraction.base import FeatureExtractorBase
  
  class FeatureExtraction(FeatureExtractorBase):
      def __init__(
              self,
              sample_rate: int,
              hop_length: int,
      ):
          self.sr = sample_rate
          self.hop_length = hop_length
  
      def get_melspectrogram(self, audio_data):
          # Return the melspectrogram of the audio
          ...
  
      @staticmethod
      def amplitude_to_db(mel_spec):
          # Return the amplitude_to_db mel spectrogram
          ...
  
      def transform(
              self,
              wave_segment: Segment
      ) -> Segment:
          # Implementation of audio feature extraction
          ...
  ``` 
* Factory Method
  The factory method, get_feature_extraction(), creates an instance of the feature extraction class. In this example,
  the FeatureExtraction class is instantiated dynamically, allowing for different feature extractors to be utilized.
  ```python
  # Factory Method for Feature Extraction
  from settings import sample_rate, hop_length
    
  def get_feature_extraction() -> FeatureExtraction:
        return FeatureExtraction(
            sample_rate=sample_rate,
            hop_length=hop_length
        )
    
    ```
## Dependency Injection (DI)

Dependency Injection (DI) is a design pattern widely employed in software development to enhance modularity and
maintainability of code. At its core, DI involves the inversion of control, where components receive their dependencies
from an external source rather than creating them internally. This external source, often called the injector or
container, manages the
instantiation and injection of dependencies into the dependent components. The primary goal of Dependency Injection is
to decouple classes
and promote a more flexible and testable codebase. By allowing dependencies to be injected dynamically, DI facilitates
the replacement of components without modifying the existing code, promoting the principles of open-closed and single
responsibility. This pattern is prevalent in various frameworks and containers across different programming languages,
contributing to the development of scalable and easily maintainable software systems. In essence, Dependency Injection
provides a powerful mechanism for achieving loose coupling and promoting the principles of object-oriented design.
