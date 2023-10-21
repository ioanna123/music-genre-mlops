# Data Preprocess

## Audio Preprocess

### What

Audio preprocessing is an essential initial step in the domain of audio analysis and processing. It involves converting
audio files to WAV (Waveform Audio File Format) to preserve audio fidelity and ensure the highest data quality. WAV is
an uncompressed audio format, which means that it retains the original audio data without any loss during compression.
This preprocessing step is pivotal as it sets the stage for accurate and reliable audio analysis and processing.

### Why

Audio preprocessing is crucial for several reasons:

* Preservation of Audio Fidelity: WAV format preserves the original audio data without any loss incurred during
  compression, maintaining the highest data quality.
* Compatibility: WAV is universally supported across various platforms, making it a suitable choice for audio-related
  applications.
* Enhanced Performance: Research by Tzanetakis and Cook (2002) has demonstrated that using uncompressed audio formats
  like WAV leads to improved performance and accuracy in Music Information Retrieval (MIR) tasks, such as music
  classification and genre recognition.

### How

* Audio Conversion Function
  To convert audio files to the WAV format, you can use the provided preprocessing function, which utilizes the "ffmpeg"
  command-line tool.

  Usage example:

    ~~~python
    from genre_classification.preprocessor.audio_preprocess import AudioPreprocess
    from settings import sample_rate
    
    audio_prep = AudioPreprocess(sample_rate=sample_rate)
    source_audio = "path/to/audio"
    
    audio_prep.self._audio_converter(source_audio_path=source_audio)
    ~~~


* Audio Streaming
  In scenarios where you deal with large audio files, such as in music genre classification, the streaming function
  becomes crucial. It addresses computational and memory limitations by processing audio data in smaller segments:
    * Standardization: The audio data is standardized to a consistent format using the _audio_converter function,
      ensuring uniformity across the dataset.
    * Sequential Processing: Leveraging the librosa_load function from the librosa library, the stream function loads
      only the specified segment of the audio file at a time, enabling sequential processing and alleviating memory
      constraints.
    * Parallelized Processing: This approach also facilitates parallelized processing, optimizing computational
      efficiency and memory utilization, making it ideal for working with vast music collections.

  Usage example:

    ~~~python
    from genre_classification.preprocessor.audio_preprocess import AudioPreprocess
    
    source_audio = "path/to/your/audio_file.mp4"
    
    audio_prep = AudioPreprocess(sample_rate=sample_rate)
    
    for streamed in audio_prep.stream(source_audio, start=0, window_duration=5):
        print(streamed)
    ~~~

## Image Dataset Preparation

### What

Image dataset preparation is the process of organizing and preprocessing image data for machine learning tasks,
especially image classification. It involves tasks such as loading image files, applying data transformations, splitting
the dataset into subsets for training, validation, and testing, and creating data loaders for efficient data processing
during training.

### Why

Image dataset preparation is critical for the following reasons:

* Data Consistency: It ensures that all images in the dataset are transformed consistently, which is essential for
  training machine learning models.
* Data Splitting: It facilitates the division of the dataset into training, validation, and test subsets, allowing for
  proper model evaluation and hyperparameter tuning.
* Efficient Data Loading: Creating data loaders enables efficient loading and processing of data during training, saving
  time and memory.

### How

The ImageDataset class provides a set of methods to streamline the image dataset preparation process.

* Use the get_transforms method to define data transformations. By default, it includes transforming images to tensors
  and normalizing them using specified mean and standard deviation values.
* The data is split into training, validation, and test subsets. You can adjust the split ratios in the settings.py
  file (e.g., SPLIT_VAL_TEST, SPLIT_TRAIN) or by modifying the ImageDataset class.
* Data loaders are created for the training and validation subsets using the DataLoader class from PyTorch. These data
  loaders facilitate batch-wise data loading and shuffling.

Here's how to use it:

~~~python
from genre_classification.preprocessor.image_dataset import ImageDataset

# Instantiate the ImageDataset Class
image_dataset = ImageDataset()

# Define Data Transformations
transform = image_dataset.get_transforms()

# Load Image Data
path_to_images = "path/to/your/images"
dataset = image_dataset.load_data(path_to_images, transform)

#   Create Data Loaders
train_dataloader, val_dataloader, test_subset, classes = image_dataset.transform(path_to_images)

~~~
