# Data Version Control (DVC)

## Data Management

### What

Efficient data versioning and administration are crucial elements in maintaining reproducibility and fostering
collaboration in ML research. This subsection explores the application of Data Version Control (DVC) in conjunction with
Amazon Simple Storage Service (S3) for managing the GTZAN dataset. The focus is on classifying musical genres,
highlighting the significance of robust data handling in ML endeavors.

### Why

At the core of connecting data-intensive projects with version control systems is DVC. Seamlessly integrating with Git,
DVC establishes a unified platform for versioning both data and code, making it suitable for a variety of datasets,
including audio files and images like spectrograms. The decision to employ DVC for GTZAN dataset management is driven by
its compatibility with Git, offering comprehensive versioning and collaboration capabilities. The collaborative
features, traceability, and scalability of DVC are essential for handling large and diverse datasets in research
settings. Integration with Amazon S3 provides researchers access to scalable object storage, ensuring effortless
scalability, cost-effective data administration, and reliable data durability. Storing the GTZAN dataset in S3
safeguards data integrity and traceability over time.

### How

With DVC, metadata about the data is versioned alongside the source code, while the original data files are
intentionally added to .gitignore. This ensures a clean version control history. The data stored in the remote S3
repository is reflected in the cache directory, with the hash value of the added files determining the cache path.
Inspection of the genres_original.dvc file reveals metadata specifying the hash, size, number of files, and path. This
comprehensive approach to data management not only ensures versioning and traceability but also optimizes storage and
transfer processes. It promotes collaboration and adherence to GitOps principles in data science projects. DVC's
lightweight and open-source nature eliminates the need for complex databases, fostering consistency through stable file
names and enabling efficient data management. Adopting DVC for data versioning and management aligns with advanced CI/CD
tools and Git workflows, enriching collaboration and knowledge exchange in the research community.

~~~yaml
outs:
  - md5: 93c37d996f5645354ebb96a940624b50.dir
    size: 1323989016
    nfiles: 1000
    path: genres_original
~~~

~~~yaml
outs:
  - md5: d51e5962bbadebae18e35019feee08b7.dir
    size: 70836023
    nfiles: 999
    path: images_original
~~~  

## Experimental Management (Machine Learning Pipelines using DVC Pipelines)

### What

In the domain of music genre classification, DVC pipelines emerge as a fundamental element, serving as a structured
framework to ensure the effectiveness, reproducibility, and organized arrangement of the machine learning (ML) process.
These pipelines are designed to capture, version, and reproduce the entire workflow, providing a systematic approach to
managing dependencies, coordinating stages, and integrating with version control systems.

### Why

The inclusion of DVC pipelines within the framework of the thesis is driven by several key advantages:

**Reproducibility**:
DVC pipelines offer a robust mechanism for capturing, versioning, and reproducing the entire music genre classification
workflow. This capability ensures that every stage, from audio preprocessing to model training, can be precisely
replicated. This contributes significantly to the credibility of research findings, providing a reliable basis for
experimentation and analysis.

**Structured Workflow**:
The structured nature of DVC pipelines allows for a clear delineation of each stage in the classification process. This
organizational clarity not only enhances the overall readability of the workflow but also facilitates seamless
coordination between different stages. This structured approach contributes to better project understanding and
collaboration.

**Version Control Integration**:
DVC integrates seamlessly with Git, offering version control for both code and data. This integration ensures that
changes in the codebase or dataset are systematically tracked, enabling precise control over the evolution of ML models.
The ability to trace and manage modifications enhances collaboration among researchers and maintains a clear history of
changes.

**Dependency Tracking**:
DVC explicitly defines dependencies between different stages of the music genre classification workflow. This ensures
that any change in the input data or code triggers the necessary downstream steps, maintaining consistency in the
workflow. Dependency tracking is crucial for understanding the impact of changes and ensuring that the entire pipeline
remains coherent and reproducible.

### How

The implementation of DVC pipelines in this thesis is characterized by a detailed code snippet, showcasing the creation
of a structured pipeline with two distinct stages: feature_extraction and train.

**Feature Extraction Stage**:
In the feature_extraction stage, a Python script is executed to generate image features from audio files, aligning with
the Audio Feature Extraction Pipeline discussed in the previous section. Dependencies, including the pipeline entry
point Python file and the directory containing audio files, are meticulously defined. Outputs specifying the path for
storing the features are also defined, ensuring that modifications in the input data trigger the execution of this
stage. This systematic definition of dependencies and outputs guarantees the reproducibility of the feature extraction
process.

**Training Stage**:
The train stage utilizes the image features generated in the previous stage (Image_data) to train a machine learning
model. Corresponding to the training pipeline from features mentioned in the previous section, the model specified as
resnet18 utilizes the specified criterion (cross_entropy) and optimizer (sdg). Checkpoints from the training process are
stored in the checkpoints directory. This stage ensures the model's continual improvement and adaptation based on the
extracted features.

The implemented pipeline, facilitated by the DVC framework, represents the Audio-Based Music Genre Classification
Pipeline detailed in the preceding sub-section. This integration seamlessly combines the feature extraction pipeline and
the model training process, orchestrating a cohesive and reproducible workflow for music genre classification.

**Workflow Structure and Reproducibility**:
By structuring the workflow using DVC pipelines, this thesis ensures a systematic and reproducible approach to music
genre classification. The clear delineation of stages, version control integration, and dependency tracking collectively
contribute to a robust and transparent ML workflow. The use of DVC pipelines enhances collaboration, facilitates
experimentation, and provides a foundation for credible research findings in the dynamic field of music genre
classification.


~~~yaml
stages:
  feature_extraction:
    cmd: python3 __main__.py create-image-features-from-audio --path_with_audios_dir Data/genres_original --path_to_image Image_data
    deps:
      - __main__.py
      - Data/genres_original
    outs:
      - Image_data
  train:
    cmd: python3 __main__.py train-using-image-features --model resnet18 --criterion cross_entropy --optimizer sdg --checkpoints_path checkpoints --images_path Image_data
    deps:
      - __main__.py
      - Image_data
    outs:
      - checkpoints
~~~
