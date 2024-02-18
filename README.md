# Designing a Scalable and Reproducible Machine Learning Workflow Thesis

## Music Genre Classification Use Case

This implementation is an integral part of my thesis conducted in collaboration with the University of West Attica,
reflecting a culmination of academic exploration and practical application in the field of machine learning.

### Introduction

In the realm of machine learning (ML), constructing end-to-end experimentation pipelines with scalability, robustness,
and reproducibility is essential for advancing ML applications. This project is dedicated to empowering Data Scientists
and ML Engineers by providing a seamless pipeline execution experience, eliminating obstacles such as downtime, hardware
unavailability, OS conflicts, or dependency issues. The overarching goal is to achieve robust execution in a highly
available environment, revisitable reproducibility, minimal manual intervention through automation, easy extendability,
and scalable capabilities for handling larger tasks concurrently. To facilitate these attributes, the project
incorporates a comprehensive toolkit encompassing Containerization/Virtualization for consistent environment management,
Monitoring experiments for provisioning necessary training information, Data/Model Tracking for tracing model and data
versions, Scalable Object Storage for secure data and model storage, and a Workflow Engine for automation, scheduling,
and monitoring. The project not only addresses the challenges of designing scalable and reproducible ML workflows but
also provides practical insights through industrial case studies, showcasing the tangible benefits of adopting such
workflows.

### Project Structure

~~~bash
|-- airflow/
|   |-- dags/
|   |-- logs/
|-- checkpoints/
|-- evidently_ai/
|-- Data/
|   |-- .dvc
|-- Dockerfile
|-- docs/
|-- dvc_pull_files.sh
|-- dvc.yaml
|-- genre_classification/
|   |-- data_model/
|   |   |-- (dir with data models used in the project)
|   |-- entrypoints.py
|   |-- feature_extraction/
|   |   |-- (dir with abstract feature extraction class, implemented class, and factory)
|   |-- __init__.py
|   |-- model/
|   |   |-- (dir with abstract model and implemented models)
|   |-- preprocessor/
|   |   |-- (dir with audio and image preprocessing classes, and the factory)
|   |-- trainer/
|   |   |-- (dir with the optimizer)
|   |-- utils/
|       |-- evaluation_metrics.py
|       |-- metadata.py
|       |-- model_selection.py
|       |-- save_load.py
|       |-- save_mel_spec_img.py
|-- __main__.py
|-- mlruns/
|-- playground.py
|-- settings.py
~~~

### Project Overview

1. airflow/: Directory containing Airflow DAGs for workflow orchestration and logs for task
   monitoring. [Internal Airflow Documentation](docs/airflow.md)

2. checkpoints/: Directory to store trained models.

3. evidently_ai/: Directory containing the code and reports from evidently. [Internal Evidently AI Documentation](docs/evidently_ai.md)

4. Data/: Directory with a DVC file to synchronize data.

5. Dockerfile: Script to build the project using Docker. [Internal Docker Documentation](docs/project_packaging.md)

6. docs/: Extended documentation for the project.

7. dvc_pull_files.sh: Script to download DVC files (models and data). [Internal DVC Documentation](docs/dvc.md)

8. dvc.yaml: DVC pipelines codebase. [Internal DVC Documentation](docs/dvc.md)

9. genre_classification/: Main package containing the core functionality of the project.

   data_model/: Directory with data models used in the project. [DataModel Overview](docs/data_model.md)

   entrypoints.py: Script with developed ML pipelines. [Pipeline Overview](docs/genre_class_pipeline_overview.md)

   feature_extraction/: Directory with an abstract feature extraction class, implemented class, and
   factory. [Feature Extraction Documentation](docs/feature_extraction.md)

   model/: Directory with an abstract model and implemented models. [DL Model Overview](docs/model_training.md)

   preprocessor/: Directory with audio and image preprocessing classes and the
   factory. [Audio Preprocess Documentation](docs/audio_preprocess.md)

   trainer/: Directory with the optimizer. [DL Model Overview](docs/model_training.md)

   utils/: Directory with scripts supporting the project (e.g., evaluation metrics, metadata handling, model selection,
   and save/load utilities).

10. __main__.py: Command-line interface to execute ML pipelines developed in the
   project. [CLI Implementation Overview](docs/cli.md)

11. mlruns/: MLflow directory holding metadata. [Internal MLFlow Documentation](docs/ml_flow.md)

12. playground.py: Script to test codebase functions.

13. settings.py: File containing project settings and global variables.

[Software Engineering patterns and principles](docs/soft_eng_tech.md)
