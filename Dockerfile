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
COPY Data/genres_original.dvc Data/genres_original.dvc
COPY Data/images_original.dvc Data/images_original.dvc
COPY genre_classification genre_classification
COPY __main__.py __main__.py
COPY settings.py settings.py
# install predictor and copy files
RUN pip install -r requirements.txt
RUN --mount=type=secret,id=aws_config --mount=type=secret,id=aws_credentials sh dvc_pull_files.sh

# Define the default command to run when the container starts
# docker run -it --rm music-genre-classif:latest python3 __main__.py train-using-image-features --model resnet18 --criterion cross_entropy --optimizer sdg --checkpoints_path checkpoints --images_path Image_data
# docker build -t music-genre-classif:latest .
