# Command-Line Interface (CLI)

### What

In the realm of ML workflows, the Command-Line Interface (CLI) serves as a potent instrument for coordinating and
executing intricate procedures. In the context of this thesis, the Click framework is employed to construct a CLI that
simplifies the implementation of music genre classification pipelines. Here, CLI refers to a text-based interface
enabling users to interact with the system by issuing commands.

### Why

A CLI is a text-based interface allowing users to interact with software and perform operations by entering commands
into a terminal or console. CLIs offer a direct and efficient means of controlling applications and processes, making
them valuable for automating tasks and managing complex workflows. Click, a Python package, simplifies the creation of
CLIs, providing a clean and concise syntax for defining command-line options, arguments, and commands. The Click
framework is particularly well-suited for this approach due to its simplicity, extensibility, and seamless integration
with Python.

**Advantages of CLI and Click**:

* Streamlined Execution:
  CLI commands facilitate a streamlined execution of complex tasks, enabling researchers and practitioners to trigger
  music genre classification pipelines with a single command. This simplicity enhances usability and reduces entry
  barriers, making it accessible for users to engage with the system efficiently.

* Reproducibility:
  Encapsulating the entire pipeline within CLI commands inherently builds reproducibility into the workflow. The defined
  options and functionalities ensure that experiments and evaluations can be consistently replicated, providing a
  reliable
  foundation for research findings.

* User-Friendly Interface:
  Click provides an intuitive and user-friendly interface for CLI commands. The clean syntax and well-defined options
  make
  it easy for users to understand and leverage the capabilities of the music genre classification system. This
  user-friendly aspect enhances the overall usability of the CLI.

* Flexibility and Extensibility:
  The modular nature of CLI commands allows for extensibility and flexibility. Additional commands can be added to the
  CLI
  to accommodate future enhancements or variations in the music genre classification pipeline. This adaptability ensures
  that the system can evolve with changing requirements and advancements in the field.

### How

In the realm of music genre classification, a specific CLI command, "train_using_original_audios," is meticulously
developed utilizing the Click framework. This command encapsulates the training pipeline for audio-based genre
classification, exhibiting key components that enhance efficiency, reproducibility, and user-friendliness.

**Options Definition**:
Various options are defined using @click.option decorators within the CLI command. These options encompass critical
aspects such as model choice, criterion, optimizer, checkpoints path, path to save featured images for training, audio
paths, number of training epochs, and others. This comprehensive definition ensures flexibility and adaptability,
allowing users to customize parameters based on specific requirements.

**Functionality Integration**:
Integrated seamlessly within the overarching ML pipeline, the "train_using_original_audios" command plays a pivotal role
by invoking the ml_entrypoints.train_tl_model_audio function. This function, positioned within the broader context of
the extended pipeline, adeptly receives the specified options and orchestrates the intricate training process for the
music genre classification model. The seamless connection ensures that the training task operates harmoniously within
the larger pipeline, contributing cohesively to the model's evolution.

**Result Handling**:
Metrics generated during the training process are systematically saved in a designated file (--path_to_save_metric).
This meticulous handling ensures that evaluation results are not only easily accessible but also reproducible,
contributing to the transparency and reliability of the research findings.

The incorporation of the Click framework for CLI commands in this thesis significantly contributes to the efficiency,
reproducibility, and user-friendliness of music genre classification workflows. CLI commands, empowered by Click, serve
as a pivotal interface for researchers and practitioners to interact with and explore the intricacies of ML processes,
providing a streamlined and accessible approach to managing and executing complex tasks in the context of music genre
classification.

~~~python
import click

import genre_classification.entrypoints as ml_entrypoints
from genre_classification.data_model.criterion import Criterion, return_criterion
from genre_classification.data_model.tl_models import TLModel
from genre_classification.trainer.optimizer import Optimizer
from genre_classification.utils.save_load import save_metrics


@click.group()
def cli():
    pass


@cli.command()
@click.option('--model', type=click.Choice([str(TLModel.vgg.value), str(TLModel.alexnet.value),
                                            str(TLModel.densenet121.value), str(TLModel.resnet34.value),
                                            str(TLModel.resnet18.value)]))
@click.option('--criterion', type=click.Choice([str(Criterion.cross_entropy.name), str(Criterion.kldiv_loss.name),
                                                str(Criterion.smooth_loss.name)]))
@click.option('--optimizer', type=click.Choice([str(Optimizer.adam.value), str(Optimizer.sdg.value),
                                                str(Optimizer.rmsprop.value)]))
@click.option('--checkpoints_path', type=click.STRING, required=True, help='Checkpoint path to save models')
@click.option('--images_path', type=click.STRING, required=True, help='Path to load the featured images for training')
@click.option('--save', type=click.BOOL, required=False, default=True,
              help='if true save the checkpoints to desired path')
@click.option('--num_epoch', type=click.INT, default=1, help='The num of epochs for training')
@click.option('--path_to_save_metric', type=click.STRING, default='metrics.json', help='The path to save the metrics')
def train_using_image_features(model: TLModel, criterion: Criterion, optimizer: Optimizer, checkpoints_path: str,
                               images_path: str, save: bool, num_epoch: int, path_to_save_metric: str):
    save_metrics(metrics=ml_entrypoints.train_tl_model_images(tl_model=model, criterion=return_criterion(criterion),
                                                              optimizer=optimizer,
                                                              checkpoints_path=checkpoints_path,
                                                              images_path=images_path,
                                                              save=save,
                                                              num_epoch=num_epoch),
                 file_name=path_to_save_metric)
~~~
