import click

import genre_classification.entrypoints as ml_entrypoints
from genre_classification.data_model.criterion import Criterion, return_criterion
from genre_classification.data_model.tl_models import TLModel
from genre_classification.feature_extraction.feature_extraction import FeatureExtraction
from genre_classification.preprocessor.audio_preprocess import AudioPreprocess
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


@cli.command()
@click.option('--model', type=click.Choice([str(TLModel.vgg.value), str(TLModel.alexnet.value),
                                            str(TLModel.densenet121.value), str(TLModel.resnet34.value),
                                            str(TLModel.resnet18.value)]))
@click.option('--criterion', type=click.Choice([str(Criterion.cross_entropy.name), str(Criterion.kldiv_loss.name),
                                                str(Criterion.smooth_loss.name)]))
@click.option('--optimizer', type=click.Choice([str(Optimizer.adam.value), str(Optimizer.sdg.value),
                                                str(Optimizer.rmsprop.value)]))
@click.option('--checkpoints_path', type=click.STRING, required=True, help='Checkpoint path to save models')
@click.option('--save_images_path', type=click.STRING, required=True,
              help='Path to save the featured images for training')
@click.option('--audio_paths', type=click.STRING, required=True, help='The audio paths eg Data/genre_originals')
@click.option('--num_epoch', type=click.INT, default=10, help='The num of epochs for training')
@click.option('--save', type=click.BOOL, required=False, default=True)
@click.option('--path_to_save_metric', type=click.STRING, default='metrics.json', help='The path to save the metrics')
def train_using_original_audios(model: TLModel, criterion: Criterion, optimizer: Optimizer,
                                checkpoints_path: str, save_images_path: str, audio_paths: str, save: bool,
                                num_epoch: int, path_to_save_metric: str):
    save_metrics(metrics=ml_entrypoints.train_tl_model_audio(tl_model=model, criterion=return_criterion(criterion),
                                                             optimizer=optimizer, checkpoints_path=checkpoints_path,
                                                             save_images_path=save_images_path, save=save,
                                                             num_epoch=num_epoch, audio_paths=audio_paths),
                 file_name=path_to_save_metric)


@cli.command()
@click.option('--path_with_audios_dir', type=click.STRING, required=True)
@click.option('--path_to_image', type=click.STRING, required=False, default='Image_data')
def create_image_features_from_audio(path_with_audios_dir: str, path_to_image: str,
                                     preprocessor: AudioPreprocess = None, feature_extractor: FeatureExtraction = None):
    ml_entrypoints.create_image_features_from_audio(path_with_audios_dir=path_with_audios_dir,
                                                    path_to_image=path_to_image, preprocessor=preprocessor,
                                                    feature_extractor=feature_extractor)


if __name__ == '__main__':
    cli()
