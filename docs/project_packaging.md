# Project Packaging

## setup.py

### What

Packaging is a critical aspect of organizing and distributing Python projects efficiently. The setup.py script, powered
by the setuptools library, plays a pivotal role in defining the project's metadata, dependencies, and distribution
specifications. This documentation aims to provide an overview of what setup.py is and why it is a valuable tool in
Python project development.
setup.py is a Python script used to configure, package, and distribute Python projects. It serves as the entry point for
setuptools, a library that enhances Python's built-in distutils to simplify the process of packaging and distributing
Python projects. The script typically contains metadata about the project, such as its name, version, description,
author, and dependencies. Additionally, it specifies how the project should be installed, including any required scripts
or entry points.

### Why

**Project Metadata**: setup.py centralizes crucial information about the project, making it easy to maintain and update.
This metadata includes the project name, version, description, author, and more, ensuring a standardized and organized
approach to project documentation.

**Dependency Managemen**t: The script provides a convenient way to declare project dependencies. By listing required
packages in the install_requires parameter, users and developers can easily identify and install the necessary
dependencies using tools like pip. This promotes a seamless and reproducible environment for the project.

**Distribution Specifications**: setup.py defines how the project should be packaged and distributed. This includes
specifying the project's packages, entry points, scripts, and other relevant details. The script empowers developers to
create distribution packages that can be easily shared, installed, and integrated into other projects.

**Installation Process**: The script facilitates the installation process of a Python project. By running commands like
python setup.py install or using pip, users can effortlessly install the project along with its dependencies. This
simplicity encourages widespread adoption and collaboration on the project.

**Script Customization**: setup.py allows for the definition of custom scripts or entry points that enhance the
project's
functionality. This is particularly useful for creating command-line tools or executable scripts associated with the
project.

**Integration with Tools**: The script integrates seamlessly with various tools in the Python ecosystem, including build
and
packaging tools. This ensures compatibility and interoperability with common development workflows and practices.

### How

The setup.py script configures the packaging for the 'music-genre-classification' Python project. Key points:

* Dependencies: Lists project dependencies (Pydantic, librosa, numpy, etc.).
* Package Configuration: Specifies project metadata (name, version, description).
* Package Inclusion: Includes package data during distribution.
* Testing: Minimal setup and test requirements.
* Package Discovery: Uses find_packages to automatically discover and include packages.
* Entry Points: Defines console scripts ('worker,' 'predictors,' etc.) for project functionalities.

Use case:

~~~python
from setuptools import setup

deps = {
    'std': [
        'pydantic==1.10.8',
        'librosa==0.9.2',
        'numpy==1.24.3',
        'ffmpeg-python>=0.2.0',
        'matplotlib==3.4.3',
        'torchvision==0.12.0',
        'torch==1.11.0',
        'dvc==2.58.2',
        'smart-open==6.3.0',
        'mlflow==1.20.0',
        'optuna==3.3.0',
        'onnx==1.15.0',
        'onnxruntime==1.17.0',
        'evidently==0.4.15'
    ]
}

setup(
    name='music-genre-classification',
    version='1.0.1',
    description='Music Genre Classification',
    install_requires=[],
    extras_require=deps,
    entry_points={
        'console_scripts': [
            'pipeline = __main__.py:cli'
        ]
    },
    packages=['genre_classification']
)
~~~

## Docker

### What

Docker is a revolutionary tool in contemporary software development, providing a robust containerization platform that
enhances the efficiency, consistency, and reproducibility of applications across diverse environments [83]. In the
context of machine learning (ML) workflows, Docker serves as a containerization solution, encapsulating the entire
runtime environment, encompassing dependencies, libraries, and configurations. This approach ensures seamless
portability and execution of ML applications across various computing environments.

### Why

The utility of Docker lies in its capability to encapsulate applications and their dependencies within isolated
containers. This encapsulation guarantees consistent behavior of the software, eliminating the common challenge of "it
works on my machine." By packaging applications alongside their runtime dependencies, Docker addresses compatibility
issues, streamlining deployment processes, and fostering collaboration among researchers, developers, and data
scientists [84]. Docker's impact extends beyond local development environments, providing a standardized and
reproducible framework for ML pipelines, thus enhancing efficiency and ensuring consistent results across different
stages of the development and deployment lifecycle.

### How

In this thesis, Docker takes center stage in managing the intricacies of music genre classification pipelines. The
provided Dockerfile employs a multi-stage building approach, starting with a base image and extending functionality
through specialized stages ('awscli' and 'dl_pipelines').

Key Features:

* Multi-Stage Building: Utilizes distinct stages ('base,' 'awscli,' 'dl_pipelines') to optimize image size and
  functionality.
* AWS CLI Integration: Incorporates AWS CLI installation in the 'awscli' stage for seamless AWS operations.
* DVC Configuration: Implements DVC configuration in the 'dl_pipelines' stage, ensuring versioned data and
  reproducibility.
* Secrets Handling: Utilizes Docker secrets for secure AWS configuration and credentials.
* Default Command: Specifies the default command to execute when the container starts, streamlining music genre
  classification tasks.

This Dockerfile encapsulates the entire pipeline, ensuring consistency, reproducibility, and ease of deployment in the
dynamic landscape of machine learning workflows 

~~~dockerfile
# syntax=docker/dockerfile:1.4

FROM python:3.8 as base

WORKDIR /usr/src/app
ARG PIP_EXTRA_INDEX_URL

RUN apt-get update \
    && apt-get install -y build-essential libpq-dev gcc --no-install-recommends \
    && apt-get install -y libblas-dev liblapack-dev --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

FROM base as awscli
WORKDIR /opt
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" &&\
    unzip awscliv2.zip

FROM base as dl_pipelines
# install dvc and pull models
RUN pip3 install dvc[s3]~=2.41.1  --no-cache-dir
ARG AWS_CONFIG_FILE=/run/secrets/aws_config
ARG AWS_SHARED_CREDENTIALS_FILE=/run/secrets/aws_credentials
COPY --from=awscli /opt/aws /opt/aws
RUN /opt/aws/install
COPY .dvc .dvc
COPY .git .git
COPY dvc_pull_files.sh dvc_pull_files.sh
COPY checkpoints checkpoints
RUN --mount=type=secret,id=aws_config --mount=type=secret,id=aws_credentials sh dvc_pull_files.sh
# install predictor and copy files
RUN pip install -r requirements.txt
COPY . .

# Define the default command to run when the container starts
# docker run -it --rm music-genre-classif python3 __main__.py train-using-image-features --model resnet18 --criterion cross_entropy --optimizer sdg --checkpoints_path checkpoints --images_path Image_data
~~~
